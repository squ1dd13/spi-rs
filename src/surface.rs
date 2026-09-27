//! Copy sample blocks within and between strided planes.
//!
//! A plane stores one component, with row starts separated by `stride` samples. A motion vector is
//! a displacement subtracted from the destination to locate the source. The caller supplies
//! current-picture storage; these helpers do not select an earlier frame or check whether a source
//! has been decoded.
//!
//! Colour uses two-dimensional rectangle bounds. Production callers pass the padded picture width
//! and height. Alpha uses a linear source address, so a source row can cross the storage row
//! boundary if the full access remains inside the allocation. Keep these rules separate when
//! changing address checks.
//!
//! Each helper validates its geometry before writing and reads the full 16 by 16 source into
//! temporary storage. Overlapping copies therefore use the original source values throughout.
//! [`copy_block16`] transfers a block between planes with separate strides. The crate page's
//! copy-method explanation connects these operations to encoded displacement units.

/// Invalid geometry or access for a decoded sample plane.
///
/// Surface operations validate both logical rectangles and their backing slice lengths before
/// copying any destination samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceError {
    /// Zero, inconsistent, or overflowing plane dimensions.
    #[error("invalid plane dimensions")]
    InvalidDimensions,
    /// A requested source or destination rectangle leaves the plane.
    #[error("plane access is out of bounds")]
    OutOfBounds,
    /// A backing slice does not cover its declared dimensions.
    #[error("plane buffer is too small")]
    BufferTooSmall,
}

/// Calculates the backing length for a strided plane.
///
/// Multiplication overflow is reported as [`SurfaceError::InvalidDimensions`].
fn required_len(stride: usize, height: usize) -> Result<usize, SurfaceError> {
    stride
        .checked_mul(height)
        .ok_or(SurfaceError::InvalidDimensions)
}

/// A whole-sample motion displacement.
///
/// Copy procedures subtract this vector from the destination to locate the source. Colour vectors
/// address two-dimensional plane coordinates; alpha vectors use its padded linear layout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Motion {
    /// The horizontal displacement in samples.
    pub x: i32,
    /// The vertical displacement in samples.
    pub y: i32,
}

/// Copies one motion-referenced colour block within a plane.
///
/// Source and destination rectangles are 16 by 16 pixels and must remain inside the explicit
/// `width` and `height` (padded picture dimensions in production). The source is at destination
/// minus `motion`. Source and destination can overlap because the full source block is read into
/// fixed scratch before any destination byte changes.
///
/// Returns [`SurfaceError`] for invalid dimensions, a short plane buffer, or an out-of-bounds
/// rectangle. It does not change the plane on validation failure.
pub fn copy_colour_motion(
    plane: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    destination_x: usize,
    destination_y: usize,
    motion: Motion,
) -> Result<(), SurfaceError> {
    if width == 0 || height == 0 || stride < width {
        return Err(SurfaceError::InvalidDimensions);
    }
    let length = required_len(stride, height)?;
    if plane.len() < length {
        return Err(SurfaceError::BufferTooSmall);
    }
    if destination_x
        .checked_add(16)
        .filter(|&v| v <= width)
        .is_none()
        || destination_y
            .checked_add(16)
            .filter(|&v| v <= height)
            .is_none()
    {
        return Err(SurfaceError::OutOfBounds);
    }
    let sx = i128::try_from(destination_x).expect("usize fits in i128") - i128::from(motion.x);
    let sy = i128::try_from(destination_y).expect("usize fits in i128") - i128::from(motion.y);
    if sx < 0
        || sy < 0
        || sx + 16 > i128::try_from(width).expect("usize fits in i128")
        || sy + 16 > i128::try_from(height).expect("usize fits in i128")
    {
        return Err(SurfaceError::OutOfBounds);
    }
    let sx = usize::try_from(sx).expect("validated source x is nonnegative");
    let sy = usize::try_from(sy).expect("validated source y is nonnegative");
    let mut scratch = [0_u8; 256];
    for row in 0..16 {
        let start = (sy + row) * stride + sx;
        scratch[row * 16..row * 16 + 16].copy_from_slice(&plane[start..start + 16]);
    }
    for row in 0..16 {
        let start = (destination_y + row) * stride + destination_x;
        plane[start..start + 16].copy_from_slice(&scratch[row * 16..row * 16 + 16]);
    }
    Ok(())
}

/// Copies one motion-referenced alpha block through the padded linear layout.
///
/// Motion is measured as the linear offset `y * stride + x`. A 16-sample source row can therefore
/// cross a visible-width boundary while remaining inside the allocated row padding, as implemented
/// by the alpha decoder. Scratch storage makes overlapping copies stable.
///
/// Returns [`SurfaceError`] for invalid dimensions, a short plane buffer, or a source or
/// destination outside the padded allocation.
pub fn copy_alpha_motion(
    plane: &mut [u8],
    stride: usize,
    height: usize,
    destination_x: usize,
    destination_y: usize,
    motion: Motion,
) -> Result<(), SurfaceError> {
    if stride == 0 || height == 0 {
        return Err(SurfaceError::InvalidDimensions);
    }
    let length = required_len(stride, height)?;
    if plane.len() < length {
        return Err(SurfaceError::BufferTooSmall);
    }
    if destination_x
        .checked_add(16)
        .filter(|&v| v <= stride)
        .is_none()
        || destination_y
            .checked_add(16)
            .filter(|&v| v <= height)
            .is_none()
    {
        return Err(SurfaceError::OutOfBounds);
    }
    let dst = destination_y
        .checked_mul(stride)
        .and_then(|v| v.checked_add(destination_x))
        .ok_or(SurfaceError::OutOfBounds)? as i128;
    let src = dst - i128::from(motion.y) * stride as i128 - i128::from(motion.x);
    let end = src + 15_i128 * stride as i128 + 16;
    if src < 0 || end > length as i128 {
        return Err(SurfaceError::OutOfBounds);
    }
    let mut scratch = [0_u8; 256];
    for row in 0..16 {
        let start =
            usize::try_from(src).expect("validated source offset fits usize") + row * stride;
        scratch[row * 16..row * 16 + 16].copy_from_slice(&plane[start..start + 16]);
    }
    for row in 0..16 {
        let start = (destination_y + row) * stride + destination_x;
        plane[start..start + 16].copy_from_slice(&scratch[row * 16..row * 16 + 16]);
    }
    Ok(())
}

/// Copies a positioned 16 by 16 block between independently strided planes.
///
/// Both horizontal rectangles must fit within their stride, and each backing slice must contain
/// all rows through the block bottom. Fixed scratch permits the slices to represent overlapping
/// logical storage if a caller arranges that safely.
///
/// Returns [`SurfaceError`] before writing when dimensions, positions, or backing lengths are
/// invalid.
pub fn copy_block16(
    source: &[u8],
    source_stride: usize,
    source_position: (usize, usize),
    destination: &mut [u8],
    destination_stride: usize,
    destination_position: (usize, usize),
) -> Result<(), SurfaceError> {
    let (source_x, source_y) = source_position;
    let (destination_x, destination_y) = destination_position;
    if source_stride < 16 || destination_stride < 16 {
        return Err(SurfaceError::InvalidDimensions);
    }
    let sh = source_y.checked_add(16).ok_or(SurfaceError::OutOfBounds)?;
    let dh = destination_y
        .checked_add(16)
        .ok_or(SurfaceError::OutOfBounds)?;
    if source_x
        .checked_add(16)
        .filter(|&v| v <= source_stride)
        .is_none()
        || destination_x
            .checked_add(16)
            .filter(|&v| v <= destination_stride)
            .is_none()
    {
        return Err(SurfaceError::OutOfBounds);
    }
    if source.len() < required_len(source_stride, sh)?
        || destination.len() < required_len(destination_stride, dh)?
    {
        return Err(SurfaceError::BufferTooSmall);
    }
    let mut scratch = [0_u8; 256];
    for row in 0..16 {
        let start = (source_y + row) * source_stride + source_x;
        scratch[row * 16..row * 16 + 16].copy_from_slice(&source[start..start + 16]);
    }
    for row in 0..16 {
        let start = (destination_y + row) * destination_stride + destination_x;
        destination[start..start + 16].copy_from_slice(&scratch[row * 16..row * 16 + 16]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlapping_colour_motion_uses_original_source_block() {
        let (w, h, s) = (32, 20, 32);
        let mut plane = vec![0_u8; s * h];
        for y in 0..h {
            for x in 0..w {
                plane[y * s + x] = (y * w + x).to_le_bytes()[0];
            }
        }
        let source: Vec<u8> = (0..16)
            .flat_map(|r| (0..16).map(move |c| ((r + 2) * w + c + 2).to_le_bytes()[0]))
            .collect();
        copy_colour_motion(&mut plane, s, w, h, 3, 2, Motion { x: 1, y: 0 }).unwrap();
        for r in 0..16 {
            assert_eq!(
                &plane[(2 + r) * s + 3..(2 + r) * s + 19],
                &source[r * 16..r * 16 + 16]
            );
        }
    }
    #[test]
    fn colour_motion_checks_visible_rectangle() {
        let mut p = vec![0; 32 * 32];
        assert_eq!(
            copy_colour_motion(&mut p, 32, 20, 32, 4, 0, Motion { x: 5, y: 0 }),
            Err(SurfaceError::OutOfBounds)
        );
    }
    #[test]
    fn alpha_motion_can_read_across_visible_row_padding() {
        let (s, h) = (24, 20);
        let mut p = vec![0_u8; s * h];
        for (i, byte) in p.iter_mut().enumerate() {
            *byte = i.to_le_bytes()[0];
        }
        let source: Vec<u8> = (0..16)
            .flat_map(|r| (16..32).map(move |c| (r * s + c).to_le_bytes()[0]))
            .collect();
        copy_alpha_motion(&mut p, s, h, 0, 1, Motion { x: 8, y: 0 }).unwrap();
        for r in 0..16 {
            assert_eq!(
                &p[(1 + r) * s..(1 + r) * s + 16],
                &source[r * 16..r * 16 + 16]
            );
        }
    }
}
