//! Decode and inspect SPI image containers.
//!
//! An image is a rectangle of pixels. Each output pixel has red, green, blue, and alpha values,
//! each from 0 to 255. Alpha controls opacity: 0 is transparent and 255 is opaque. The returned
//! `image::RgbaImage` uses straight alpha, which means that RGB values are not multiplied by
//! alpha. RGB values remain present when alpha is zero.
//!
//! This page describes the rules implemented by this decoder. The repository does not contain an
//! independent SPI specification. Unknown header fields, unsupported methods, and decoder limits
//! are identified below. No expansion of the name SPI or relationship to another codec is assumed.
//!
//! # Reading order
//!
//! Read [Using the API](#using-the-api), then [Container and records](#container-and-records),
//! [Picture geometry](#picture-geometry), and [Block methods](#block-methods). Continue with
//! [Prediction](#prediction), [Coded values and reconstruction](#coded-values-and-reconstruction),
//! and the [worked picture](#worked-picture). The [state table](#state-lifetimes) and
//! [implementation map](#implementation-map) connect these explanations to maintenance work.
//! Detailed contracts are next to the functions in the Rust source. Build with
//! `--document-private-items` to read the internal module pages; decoder modules remain private.
//!
//! # Using the API
//!
//! [`read_header`] inspects a complete container. It checks both outer section lengths, the
//! picture marker, and non-zero dimensions. It does not parse records or check whether the colour
//! mode and selector can be decoded. A successful inspection is therefore not a successful decode.
//!
//! [`decode`] uses the default limits: 256 MiB of input and an accounted memory limit of 512 MiB.
//! [`decode_with_limits`] accepts explicit limits. Both return visible pixels in row order, with
//! no padding. The input must be a complete container, not a bare picture header, record, or
//! colour stream.
//!
//! ```no_run
//! use spi::{DecodeLimits, decode_with_limits, read_header};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let input = std::fs::read("picture.spi")?;
//!     let info = read_header(&input)?;
//!     println!("{} by {} pixels, selector {}", info.width, info.height, info.selector);
//!     let image = decode_with_limits(&input, DecodeLimits::default())?;
//!     assert_eq!(image.dimensions(), (info.width, info.height));
//!     Ok(())
//! }
//! ```
//!
//! Decoding accepts colour mode 0 and selectors 2, 4, and 5. Selector 2 has colour only; its
//! output alpha is 255. Selectors 4 and 5 both add an alpha stream after colour in each record.
//! The decoder makes no further distinction between these two selectors.
//!
//! # Container and records
//!
//! The container has two length-prefixed sections. A length prefix is a byte count stored before
//! the bytes that it measures. Outer lengths are unsigned 32-bit little-endian integers: the least
//! significant byte comes first. Let `P` be the picture-section length and `R` the record-section
//! length.
//!
//! | Byte offset from container start | Size in bytes | Contents |
//! | --- | --- | --- |
//! | 0 | 4 | `P`, excluding this length field |
//! | 4 | `P` | Picture section |
//! | `4 + P` | 4 | `R`, excluding this length field |
//! | `8 + P` | `R` | Record section |
//!
//! The container must end at `8 + P + R`. Extra container bytes are rejected. The picture section
//! must have at least 16 bytes. Extra bytes inside that section are accepted and ignored.
//! Multi-byte picture fields are big-endian: the most significant byte comes first.
//!
//! | Byte offset from picture-section start | Size in bytes | Meaning |
//! | --- | --- | --- |
//! | 0 | 2 | Marker `AA 01` |
//! | 2 | 4 | Unknown word |
//! | 6 | 1 | Colour mode |
//! | 7 | 2 | Unknown value |
//! | 9 | 2 | Unknown value |
//! | 11 | 2 | Visible width in pixels, non-zero |
//! | 13 | 2 | Visible height in pixels, non-zero |
//! | 15 | 1 | Output selector |
//!
//! The record section contains consecutive records, with no separator or outer record count. The
//! parser counts records from their lengths. Each record starts with marker `AA 02`, followed by a
//! big-endian 32-bit total byte length. This length includes the marker and length field. It must
//! be at least 14 bytes: 6 framing bytes and 8 syntax-header bytes. The colour stream starts at
//! byte 14 from the record start.
//!
//! All bit fields are read most-significant bit first within each byte. The syntax-header table
//! uses bit offsets from byte 6 of the record. Add 48 to obtain complete-record bit offsets.
//!
//! | Syntax-header bit offset | Width in bits | Meaning |
//! | --- | --- | --- |
//! | 0 | 8 | Unknown byte B |
//! | 8 | 16 | Record index |
//! | 24 | 8 | Unknown byte A |
//! | 32 | 1 | Unknown flag |
//! | 33 | 2 | Unknown value |
//! | 35 | 8 | Unknown byte C |
//! | 43 | 8 | Unknown byte D |
//! | 51 | 8 | Unknown byte E |
//! | 59 | 4 | Must be zero |
//! | 63 | 1 | End flag; must be zero |
//!
//! Record framing can be parsed without checking the indices. Decoding requires indices 0, 1, 2,
//! and so on, in section order. Record syntax is bounded by the declared record length. When alpha
//! is present, its start is the colour endpoint rounded upwards to a multiple of eight bits. No
//! separate alpha offset or length is stored. Alignment bits need not be zero. With colour only,
//! the rounded endpoint must reach the record end. With alpha, fewer than eight unused final bits
//! are accepted; one or more unused complete bytes are rejected.
//!
//! # Picture geometry
//!
//! A sample is one component value at one pixel position. A plane is an array of samples for one
//! component. Internal byte colour planes have green, blue, red order. The decoder processes
//! square blocks of 16 by 16 pixels. Predictive methods can divide these blocks into smaller
//! squares of width 4 or 8.
//!
//! Let `W` and `H` be the visible width and height. The padded grid has `Bx = ceil(W / 16)` block
//! columns and `By = ceil(H / 16)` block rows. Here, `ceil` means round upwards to an integer.
//! Colour planes have a stride of `16 * Bx` samples: stride is the distance between consecutive
//! row starts. All 256 samples of every block are reconstructed, including samples to the right
//! and below the visible rectangle. Only output composition removes this padding.
//!
//! A record slice is a horizontal group of block rows. It is not a Rust slice, which is a view of
//! array elements. For `N` records, the decoder derives the nominal record height `S = ceil(By /
//! N)` in block rows. There is no parsed slice-height field. Record `i` starts at block row `i *
//! S` and contains `min(S, By - i * S)` rows. A start outside the picture is rejected. Thus, even
//! `N <= By` does not guarantee valid coverage: five block rows with four records give starts 0,
//! 2, 4, 6, and the last start is invalid.
//!
//! Within each record, blocks run from left to right, then from top to bottom. Block `(bx, by)`
//! has pixel origin `(16 * bx, 16 * by)`, where `by` is a picture-wide block row. A sample `(x,
//! y)` inside that block is at `(16 * bx + x, 16 * by + y)`. Temporary record planes use a row
//! relative to the record; persistent planes use the picture row.
//!
//! For a 17 by 33 picture with two records, the padded size is 32 by 48. Each cell below is a 16
//! by 16 block. The final column contributes only one visible pixel per row, and the final block
//! row contributes only one visible row.
//!
//! ```text
//!               columns 0..15   columns 16..31
//! record 0      +-------------+-------------+
//! rows 0..15    | block 0     | block 1     |
//! rows 16..31   | block 2     | block 3     |
//!               +-------------+-------------+
//! record 1      | block 0     | block 1     | rows 32..47
//!               +-------------+-------------+
//! visible rectangle: x = 0..16, y = 0..32
//! ```
//!
//! Alpha uses stride `16 * (Bx + 2)`, which adds 32 samples after the padded block width. Its
//! persistent plane has `16 * By` rows. Its temporary record plane has an additional 16 rows.
//! These allocation details are decoder choices, not additional visible pixels or fields in the
//! file.
//!
//! # Block methods
//!
//! A block method specifies how to reconstruct a 16 by 16 block. A prediction mode specifies how
//! to estimate samples inside a predictive block. A direction-table index selects a displacement
//! formula. These are three distinct numbers.
//!
//! | Encoded bits | Method number | Colour action | Alpha action |
//! | --- | --- | --- | --- |
//! | `1` | 0 | Copy default neighbour | Copy default neighbour |
//! | `01` | 1 | Copy with displacement | Copy with displacement |
//! | `0000` | 2 | Run-length-coded byte values, optionally with vertical residuals | Unsupported |
//! | `0001` | 3 | Prediction plus transformed coefficients | Prediction plus directly applied coded residuals |
//! | `0010` | 4 | Palette | Unsupported |
//! | `0011` | 5 | Raw byte planes | Raw alpha bytes |
//!
//! ## Copy methods
//!
//! The default neighbour is the block above when the destination is in the first block column. In
//! every other column it is the block to the left. At the top-left picture block, the default
//! source is outside storage and decoding fails.
//!
//! Displacement-based copying is called motion copying in the code. It reads the current picture
//! storage, not an earlier animation frame. The source is the destination minus `(dx, dy)`. A
//! leading motion bit of one means `(0, 16)`. Otherwise, each component has `z` zero bits, a
//! terminating one, then `z` suffix bits `s`. Calculate `c = 2^z + s`; an even `c` gives `c / 2`,
//! and an odd `c` gives `-(c / 2)`, with integer division. Multiply by 16 to obtain samples.
//! Prefixes with 31 zeroes are rejected. Colour rejects multiplication overflow; alpha uses
//! saturating multiplication.
//!
//! Colour checks the complete source rectangle against the padded picture width and height. Alpha
//! instead calculates the linear source address `destination_address - dy * stride - dx`, then
//! checks the allocation bounds through the last copied row. An alpha source row can cross a row
//! boundary in storage. Neither copy path checks whether source samples have already been decoded.
//! Unwritten picture storage starts at zero. Both paths copy through a temporary 256-byte array so
//! overlap does not change later source reads.
//!
//! ## Byte values, palettes, and raw samples
//!
//! Colour method 2 reads one prediction flag and three run-length-coded planes. A run represents
//! repeated values. A zero flag enables vertical residual reconstruction; a one flag uses the
//! decoded bytes directly. With prediction, byte `b` gives signed change `(b >> 1) XOR -(b & 1)`,
//! then that change is multiplied by four. Add it to the prepared top sample in the first row, or
//! to the reconstructed sample in the preceding row. If the result exceeds 257, subtract 260 once;
//! if it is below -2, add 260 once. Finally clip it to 0..255. This is not a general modulo
//! operation.
//!
//! A palette is a list of green-blue-red triplets. Each pixel selects an entry by its index. The
//! palette starts empty for each record. Marker zero reuses a palette with a non-zero index width.
//! Marker one reads a variant bit and a four-bit width `p`. If `p` is zero, three bytes define one
//! entry and fill the block; the variant is ignored. Otherwise a `p`-bit value plus one gives the
//! new entry count. Variant one replaces entries; variant zero retains the existing prefix of
//! entries and reads only additional entries, or truncates the list when the new count is smaller.
//! The same `p` sets the width of coded indices. Indices are narrowed to their low byte and
//! checked against the entry count.
//!
//! Raw method 5 first skips to the next byte boundary without checking the skipped bits. Colour
//! then reads 256 green bytes, 256 blue bytes, and 256 red bytes, each in row order. Alpha reads
//! 256 bytes in row order. Raw storage does not use prediction or a palette.
//!
//! # Prediction
//!
//! Prediction estimates samples from neighbouring samples in the same picture. This is intra
//! prediction: references come from within the picture. A reference sample is a value supplied to
//! the estimate. A residual is a signed change added to that estimate. Predictive colour
//! reconstructs residuals from coefficients; predictive alpha applies its decoded array directly,
//! with extra accumulation for two modes.
//!
//! Let `(x, y)` be the zero-based column and row inside a square of width `w`. The block's
//! immediate top samples lie at `(x, -1)`, left samples at `(-1, y)`, and corner at `(-1, -1)`.
//! Extensions continue beyond the top-right and bottom-left corners. They can contain real samples
//! or replacement values.
//!
//! ```text
//! corner   top[1] top[2] ... top[w] top[w+1] ... top[2*w]
//! left[1]  (0,0)  (1,0) ...
//! left[2]  (0,1)  (1,1) ...
//!  ...
//! left[w]
//! left[w+1] ... left[2*w]   extension below the left edge
//! ```
//!
//! This diagram uses corner-inclusive banks: index 0 is the corner and index 1 is the immediate
//! neighbour. The byte `PredictionReferences` type instead has a separate `top_left` field and
//! immediate neighbours at `top[0]` and `left[0]`. The planar `SignedReferences` type also starts
//! with immediate neighbours at index 0; its constructor drops the bank corner, copies 32 values,
//! and leaves element 32 zero. A bank is a retained array of reference samples. Do not apply one
//! bank's index convention to a different representation.
//!
//! Colour prepares banks before methods 2 and 3. On the first local row of every record, the first
//! block uses neutral byte references of 128. Later blocks on that row derive the top bank from
//! the first left sample, and extend the left bank with its last sample. They do not read the
//! physical row above the record. On later rows, availability flags determine which edges can be
//! used. Missing edges are filled from an available edge or corner; missing extensions repeat an
//! endpoint. With no available neighbour, both edges are 128. Colour's top extension stops at
//! padded width.
//!
//! Alpha prepares banks before every block and updates them after successful reconstruction. At
//! the first block of every record, both banks are 128. For later blocks on that first local row,
//! the top bank repeats the previous block's saved top-right sample. Later rows use prepared
//! bottom edges from the preceding row. Alpha's top extension repeats the last sample at the
//! visible width, not the padded width. Smaller blocks can use samples reconstructed earlier
//! inside the containing 16 by 16 block. Their edge arrays persist across slots because some
//! constructors update only part of an edge.
//!
//! ## Modes and filtering
//!
//! “Signed” in predictor names means that samples use `i16`. It does not specify the direction's
//! sign. Colour's first predictive component is luma-like, in 0..255; its other two components
//! store colour differences with a bias of 256, in 0..511 after reconstruction. Alpha references
//! remain byte values even when converted to `i16` for a shared predictor.
//!
//! | Prediction mode | Operation |
//! | --- | --- |
//! | 0 | Repeat the top sample down each column |
//! | 1 | Repeat the left sample across each row |
//! | 2 | DC prediction: an edge-average estimate, with boundary adjustments |
//! | 3 | Diagonal: select an edge by the difference between column and row |
//! | 4 | Top-oriented angular prediction, step -13/32 sample per row |
//! | 5 | Top-oriented angular prediction, step +13/32 sample per row |
//! | 6 | Follow the advancing top diagonal: colour and wider alpha select a prepared sample; byte alpha at width 4 averages three samples, with an endpoint exception |
//! | 7 | Left-oriented angular prediction, step -13/32 sample per column |
//! | 8 | Left-oriented angular prediction, step +13/32 sample per column |
//! | 9..16 | Additional angular directions, listed below |
//! | 17 | Planar: blend horizontal and vertical ramps |
//!
//! The colour dispatcher filters references for width 4 modes 3, 6, 9; width 8 modes 3 through 8;
//! and width 16 modes 3 through 16. The filter replaces each interior sample `b`, with neighbours
//! `a` and `c`, by `(a + 2*b + c + 2) >> 2`. The corner becomes `(left[1] + 2*left[0] + top[1] +
//! 2) >> 2` in both output banks. The endpoint at index `2*w` is copied unchanged, and larger
//! indices are not written. All these sums use `i32`. The signed right shift divides by four with
//! rounding towards negative infinity after the offset is added.
//!
//! Colour mode 8 at width 4 uses the original left samples. Widths 8 and 16 use filtered samples.
//! The helper parameter named `filtered_left` does not establish a filtering requirement for all
//! callers. Alpha width 4 uses byte predictors for modes 0, 1, 2, 4, 5, 6, 7, 8, 17. For byte
//! alpha mode 6, let `j=x+y` in the corner-excluding top array. If `j >= 2*w-2`, use `top[2*w-1]`;
//! otherwise use `(top[j]+2*top[j+1]+top[j+2]+2)>>2`. Colour mode 6 instead selects
//! corner-inclusive `top[x+y+2]` after the dispatcher's filter. Other alpha cases use signed
//! predictors with the same filtering selection. Production alpha does not enable the byte
//! helper's special first-row flags: record-edge replacement has already occurred in its banks.
//!
//! For DC, the base is `(w + sum(top neighbours) + sum(left neighbours)) / (2*w)`. Colour and
//! signed alpha adjust the first row and column with width-dependent weights. Byte alpha at width
//! 4 has the same width-4 adjustment. For planar prediction, the horizontal ramp joins the current
//! left sample to the final top sample inside the width; the vertical ramp joins the current top
//! sample to the final left sample. Add the two ramps and `w`, then divide by `2*w`. The far
//! sample is corner-excluding index `w-1`, not the first extension sample.
//!
//! | Mode | Direction-table index | Main edge | Step in units of 1/32 sample |
//! | --- | --- | --- | --- |
//! | 9 | 17 | Left | +32 |
//! | 10 | 2 | Top | -21 |
//! | 11 | 4 | Top | -5 |
//! | 12 | 6 | Top | +5 |
//! | 13 | 8 | Top | +21 |
//! | 14 | 10 | Left | -21 |
//! | 15 | 12 | Left | -5 |
//! | 16 | 14 | Left | +5 |
//!
//! An angular step is a fixed-point displacement, not an angle in degrees. For a left-oriented
//! positive step 13, calculate `d = (x+1)*13`, `i = y + (d >> 5) + 1`, and `f = d & 31`.
//! Corner-inclusive references give prediction `((32-f)*left[i] + f*left[i+1] + 16) >> 5`. The
//! weights sum to 32. A negative step can project before the corner; the decoder builds negative
//! reference indices from the other edge. The general predictor starts an inverse accumulator at
//! 128, adds 1638, 630, 390, or 256 for absolute steps 5, 13, 21, or 32, and selects side index
//! `min(accumulator >> 8, 32)`. Dedicated modes 4 and 7 have width-specific extensions; their
//! contracts describe those limits.
//!
//! For a complete width-4 mode-8 example, use corner-inclusive `left[1..=6] =
//! [100,132,164,196,228,255]`. No filter is applied. At `(0,0)`, `d=13`, `i=1`, and `f=13`:
//! `(19*100 + 13*132 + 16) >> 5 = 113`. At column 2, `d=39` advances the integer index and leaves
//! fraction 7. The complete predicted block is:
//!
//! ```text
//! 113 126 139 152
//! 145 158 171 184
//! 177 190 203 216
//! 209 222 234 245
//! ```
//!
//! # Coded values and reconstruction
//!
//! Entropy decoding converts variable-length bit sequences to values. A symbol is one decoded
//! choice, such as a prediction mode or a coefficient's magnitude and position change. A coding
//! context is stored information that changes how the next choice is interpreted. Here it includes
//! neighbouring mode symbols and the contextual bit accumulator. It does not imply a standard
//! arithmetic coder.
//!
//! A component is one value per pixel, such as alpha or a transformed colour component. An
//! argument identifies a region of that component inside a 16 by 16 block. Subdivided colour and
//! predictive alpha use four 8 by 8 arguments, ordered top-left, top-right, bottom-left,
//! bottom-right. Each argument contains one width-8 slot or four width-4 slots in the same order.
//! A slot is one square coefficient and prediction region. Undivided colour instead uses one
//! argument with one width-16 slot. Layout slot count is distinct from the number of slots with
//! coded coefficients.
//!
//! Colour method 3 reads a profile bit: zero selects alternate scaling, one selects normal
//! scaling. Each of its three components then has a subdivision bit. A set bit selects four
//! arguments, each with a further bit: zero selects one width-8 slot; one selects four width-4
//! slots. An unset subdivision bit selects width 16 directly. Alpha method 3 requires its initial
//! bit to be one and then reads four arguments, each with the same width-selection rule: zero
//! gives one width-8 slot and one gives four width-4 slots. Its zero initial bit is rejected.
//!
//! Mode history stores symbols on a grid of four-pixel cells. It is not pixel history. A mode code
//! can select the left or delayed upper mode, or read an explicit mode. Initial history is mode 2.
//! Stored symbol 17 is treated as mode 2 when selecting a neighbour, but remains planar mode 17
//! when passed to prediction. Layout decoding updates this history before coefficient decoding, so
//! an error can leave changed history.
//!
//! A prefix of `z` zero bits followed by one can specify a suffix of `z` bits. The unsigned index
//! is `(2^z - 1) + suffix`. For example, `1` represents 0, `010` represents 1, and `011`
//! represents 2. Coefficient counts use one more than that index. Coefficient symbols add a sign
//! bit and use a short-code table or a long-code partition to obtain magnitude and scan delta. A
//! scan delta skips positions in the chosen scan order. The cursor starts at zero, advances by the
//! delta, places the signed value, then advances by one.
//!
//! Coefficients are stored as a row-major square, but bits need not visit that square in row
//! order. Colour uses a diagonal scan. Alpha can use diagonal, row, or column scans according to
//! the prediction mode. For width 4, diagonal positions begin 0, 4, 1, 2, 5, 8: scan index 1 thus
//! refers to row 1, column 0. A coefficient describes a weighted contribution to a residual
//! pattern in colour reconstruction. It is not generally the residual at the same array index.
//!
//! The activity syntax uses lookup tables. Width 4 reads an unsigned index into a 64-entry table;
//! colour and alpha use different tables. Widths 8 and 16 use the zero-prefix width itself as an
//! index into a shared table, without an index suffix. One-slot layouts test bit 2 of the result;
//! four-slot layouts test bits 5, 4, 3, 2. An inactive slot has zero residual. An active slot
//! reads a coefficient count and that many signed symbols. One coefficient at scan position zero
//! selects `ConstantResidual`; other active arrangements select `FullBlockTransform`. These names
//! are callback selectors in the implementation, not function pointers or additional bits in the
//! stream.
//!
//! ## Colour reconstruction and conversion
//!
//! Colour scales coefficients with integer weights, an operation called dequantisation here, then
//! applies an inverse transform to produce signed residual samples. A separable transform first
//! combines values along columns and then along rows. Intermediate narrowing and rounding are part
//! of this implementation. Widths 4 and 8 use integer additions, shifts, and profile-specific
//! weights; width 16 uses a fixed integer matrix. The small full-transform paths transpose their
//! result when they write the residual array. Do not replace these calculations with a
//! floating-point transform.
//!
//! In a transform, DC means the coefficient at position zero that gives a constant contribution
//! across a block. DC prediction is a separate operation: it averages reference edges. Normal
//! width-4 constant reconstruction scales coefficient `c` as `(288*c + 1) >> 1`, narrows to `i16`,
//! then calculates `(scaled + 32) >> 6` and fills all 16 residual samples. Thus `c=5` gives 720
//! and then 11. The production width-4 full-transform selector instead uses normal DC weight 144,
//! direct multiplication, two transform passes, and a transposed final result; a lone DC value 1
//! on that path gives 2. Alternate width-4 scaling uses weight 160 for DC without the preliminary
//! shift. Width-8 normal scaling uses `(weight*c + 4) >> 3`; alternate scaling uses `(weight*c +
//! 2) >> 2`. Its full transform adds 32 to element zero with `i16` wrapping before the passes and
//! shifts the final pass right by six. Width 16 uses scale 144 normally or 160 for the alternate
//! profile, rounding offsets 128 before a shift of eight and 2048 before a shift of twelve.
//!
//! Add each residual to its prediction. Clip the first transformed component to 0..255 and the
//! other two to 0..511. Reference preparation derives these components from byte planes `G`, `B`,
//! `R` as follows. The right shifts are signed arithmetic shifts; they round negative halves
//! downwards.
//!
//! ```text
//! V = R - B
//! M = B + (V >> 1)
//! U = G - M
//! L = M + (U >> 1)
//! stored predictive components = L, U + 256, V + 256
//! ```
//!
//! `L` is called luma-like because it combines the three channels. This is not a claim that it
//! follows a named colour-space standard. The two biases make signed colour differences
//! non-negative in the predictive representation.
//!
//! For reconstructed components `L`, `U+256`, `V+256` in their clipped ranges, byte conversion is
//! equivalent to these operations in this order:
//!
//! ```text
//! G = clip(L + ceil(U/2), 0, 255)
//! B = clip(L - floor(U/2) - floor(V/2), 0, 255)
//! R = clip(B + V, 0, 255)
//! ```
//!
//! The actual code uses unsigned wrapping words, logical shifts, and signed low-halfword
//! extraction to obtain these results. `R` uses the already clipped `B`. Local variables named
//! `red`, `green`, and `blue` in that code write stored planes 0, 1, 2 respectively; those planes
//! still mean G, B, R. Output composition writes planes 2, 0, 1, then alpha.
//!
//! ## Alpha reconstruction
//!
//! Alpha uses the same component syntax machinery but does not call the colour inverse transforms.
//! Its decoded row-major values act directly as residuals. Mode 0 first accumulates them down each
//! column; mode 1 accumulates them across each row. Those additions wrap in `i16`. Other modes use
//! the values as decoded. An inactive callback contributes zero; both active callback kinds use
//! the array without constant expansion. The predictor first clips its result to a byte, then
//! alpha adds the residual and clips again to 0..255. Alpha does not multiply or clear the RGB
//! channels.
//!
//! For example, take a width-4 alpha slot in mode 1 with predicted value 250 at every sample.
//! Suppose its first residual row is `[3,4,-2,-260]`. Horizontal accumulation gives
//! `[3,7,5,-255]`; addition to prediction gives `[253,257,255,-5]`, and clipping gives
//! `[253,255,255,0]`. These values are illustrative decoded inputs, not an encoded fixture. If an
//! active slot instead contains only value 5 at array position zero, mode 1 carries that 5 across
//! the first row through accumulation. The other three rows remain at prediction because their
//! residuals are zero. It does not fill the complete square as a colour constant transform would.
//! An inactive slot contributes zero residual everywhere.
//!
//! ## Runs and palette indices
//!
//! Byte run-length decoding reads literals using a unary prefix and two remainder bits. Prefixes
//! below 21 give `4*prefix + remainder`; prefix 21 with remainder 0..2 gives 84..86; other cases
//! read a raw byte. When the preceding two literals match, a run code gives `(2^z - 1) + suffix`,
//! narrowed to eight bits. The decoder writes that many repeats, then consumes a further literal
//! even if the run has already reached the requested output length. Logical production can
//! therefore exceed 256; it is bounded at 512, while only the requested samples are written.
//!
//! Palette index decoding reads a fixed-width literal and an unsigned run-control index. Run
//! control zero advances one pixel; a positive control repeats the literal for `control + 1`
//! pixels. Exactly 256 indices are produced. Runs beyond the block are rejected.
//!
//! # Bit readers and positions
//!
//! Ordinary colour syntax uses a checked cursor over the record payload. A read of 1..32 bits
//! either succeeds and advances, or fails without advancing that read. Earlier successful reads
//! remain consumed. Contextual syntax uses a 32-bit accumulator with valid bits aligned to the
//! left. Initialisation at bit `b` requires four bytes starting at `floor(b/8)`, loads them in
//! big-endian order, and discards `b mod 8` leading bits. This lookahead requirement can reject a
//! short stream even when a few syntax bits remain.
//!
//! The accumulator's byte offset points after the last loaded byte. Its logical position is
//! `byte_offset*8 - available_bits`. Buffered bytes are therefore not all consumed bytes. A refill
//! replaces the accumulator with up to four subsequent bytes; it does not append. Entropy routines
//! explicitly retain and combine fragments when a value crosses a refill. Contextual reads can
//! partly advance before failure. Failed initialisation leaves the previous state unchanged.
//!
//! Predictive colour and palettes with a non-zero prefix initialise a contextual reader at the
//! ordinary cursor. On success the ordinary reader consumes bits up to the contextual endpoint. On
//! failure it can remain at the entry position even though contextual syntax and history have
//! advanced. Both colour readers use payload-relative positions; the colour layer adds 112 bits
//! for the 14-byte record headers. Alpha uses the complete record as its reader input, so its
//! positions already have the correct origin. Raw alpha resets the accumulator after its aligned
//! 256 bytes.
//!
//! # Worked picture
//!
//! Consider the illustrative 17 by 33 picture above, colour mode 0, selector 2, with two records
//! indexed 0 and 1. This is a conceptual sequence, not an encoded fixture. The outer lengths
//! locate the picture and records. The dimensions produce a 2 by 3 padded block grid. Two records
//! give nominal height two block rows: record 0 covers rows 0..31, and record 1 covers rows
//! 32..47.
//!
//! Suppose record 0 starts with a raw block whose G, B, R values are 20, 30, 40 throughout. Its
//! method bits are followed by byte alignment and three 256-byte planes. The next block uses
//! default spatial copying and obtains the same values from the left. The first block of the next
//! row copies from above, and the next copies from the left. Record 1 can also start with a
//! spatial copy from the preceding record's stored picture row. That dependency remains valid even
//! though predictive top-bank preparation would use record-first-row replacements.
//!
//! Composition selects only columns 0..16 and rows 0..32. Each visible output pixel is `[40, 20,
//! 30, 255]`. There are 561 pixels, or 2244 output bytes. The samples outside the visible
//! rectangle remain relevant during block reconstruction, but are absent from the result.
//!
//! For a nontrivial predictive path, suppose an interior colour block selects method 3, normal
//! profile, and subdivision. Prepared byte edges are converted to `L`, `U+256`, `V+256` banks.
//! Within one width-8 argument, the width bit selects four width-4 slots. Mode decoding selects
//! mode 8 for a slot with left values from the numerical example above. Activity syntax marks that
//! slot active. A count code of `1` gives one coefficient. The coefficient code `0001 0110` has
//! prefix width 3, payload 6, positive sign, and short-table entry 10: magnitude 5, delta 0. It
//! writes coefficient 5 at position zero and selects constant reconstruction.
//!
//! That slot's normal residual is 11 everywhere. Add it to the mode-8 prediction. For the first
//! component, the first row becomes `[124, 137, 150, 163]`, and the final sample clips from 256 to
//! 255. The reconstructed samples enter the 16 by 16 component workspace and can supply references
//! for later slots. Mode symbols have already entered record-local coefficient history. After all
//! arguments and components finish, conversion writes G, B, R to the temporary record planes and
//! persistent picture planes. If one reconstructed position has `L=124` and both biased
//! differences 256, it produces G=B=R=124. With selector 2, its visible RGBA value is
//! `[124,124,124,255]`. With alpha present, the separately reconstructed alpha value replaces 255.
//!
//! # State lifetimes
//!
//! | State | Initial data | Update and reset | Lifetime |
//! | --- | --- | --- | --- |
//! | Colour picture planes | Zero bytes, padded G/B/R | Written by each decoded block; no record reset | Picture |
//! | Colour availability | Zero per padded block | Set after the method code, before reconstruction; used for bank preparation, not motion validation | Picture |
//! | Colour byte banks | Zero, except unused bank 8 filled with 128 | Prepared before residual and predictive blocks; first local row replaces edges | Picture object |
//! | Six colour transformed banks | Zero `i16` values | Rebuilt from byte banks: top L/U/V, then left L/U/V, with biases above | Picture object |
//! | Alpha picture plane | Zero bytes | Written by successful blocks and motion copying | Picture |
//! | Alpha left/top and next-row banks | 128 | Prepared before blocks; right and bottom edges saved after success; top banks rebuilt after each row | Picture object |
//! | Alpha row-bottom samples | 128 across padded width | Each successful block writes its bottom edge; each completed row supplies next-row banks | Picture |
//! | Alpha availability | Zero | Completed row becomes previous-row availability | Picture |
//! | Alpha previous/current availability arrays | Zero | Copy persistent availability into previous and clear current at each row; mark current after successful blocks | One record |
//! | Alpha saved block edge | 128 per block column | Top-right sample saved after every successful block | Picture |
//! | Coefficient-mode histories | Mode 2; neutral flags true | Updated by layout decoding; row transition retains delayed values and resets active cells | New each record; colour allocated lazily |
//! | Colour palette | Empty | Replace, extend, truncate, or reuse on palette blocks | New each record |
//! | Record output planes | Zero | Blocks write padded samples; visible samples copied to RGBA | One record |
//! | Component and residual arrays | Zero | Slots write their active squares; edge constructor arrays can retain values between slots | One component argument or block |
//! | Final RGBA buffer | Zero | Visible rows written after both streams finish | One decode call |
//!
//! Persistent storage makes records dependent on earlier records. This does not mean that every
//! history survives a record boundary. In particular, palettes and coefficient-mode histories do
//! not survive. A failed decode does not return a partial image, but private helpers can have
//! changed buffers, availability, banks, or cursors before the error. Their contracts identify
//! important partial effects.
//!
//! # Errors and resource limits
//!
//! [`DecodeError::kind`] returns stable categories for parsing, unsupported output or colour
//! modes, unsupported methods, bitstream syntax, coefficients, prediction, motion, coverage,
//! resource limits, and allocation failure. A recognised but unimplemented alpha method is
//! unsupported syntax. A missing field or invalid code is malformed syntax. Some block layers
//! convert low-level motion or prediction failures to their stream's bitstream category; not every
//! geometric failure is returned as `ErrorKind::Motion`.
//!
//! [`ErrorContext`] uses zero-based record and record-local block numbers. Colour component
//! context is `plane*4 + argument`; alpha component context is the argument number. Bit positions
//! are relative to the complete record, not the container. Some failures report the block's start
//! and others report the current cursor. Context wrappers are nested rather than merged; inspect
//! the error chain for the most specific position. No position means the layer did not establish
//! one.
//!
//! Limits are inclusive. Overflow in byte accounting reports a requested size of `usize::MAX`. The
//! initial memory estimate includes RGBA output, padded colour storage, optional alpha storage,
//! record descriptors, and a 4096-byte allowance per padded block for contexts. Later checks use
//! allocation capacities and palette growth. The live heap account excludes the caller's input
//! allocation, allocator overhead, stack storage, and PNG encoding.
//!
//! There is a known accounting discrepancy in the current implementation: coefficient history
//! charges `ceil(capacity/8)` for its `Vec<bool>`, although Rust stores one byte per `bool`
//! element. The initial context allowance is conservative, but the live-capacity calculation must
//! not be described as an exact upper bound. This documentation records the discrepancy without
//! changing decoder behaviour.
//!
//! # Implementation map
//!
//! | Source | Responsibility |
//! | --- | --- |
//! | `src/format.rs` | Borrow sections and records; validate markers, lengths, fixed fields |
//! | `src/decoder.rs` | Derive geometry, check resources, sequence records, compose RGBA |
//! | `src/colour/blocks.rs` | Read colour methods, raw samples, byte residuals, palettes, and copies |
//! | `src/colour/history.rs`, `src/colour/references.rs` | Store picture samples; prepare byte and transformed edges |
//! | `src/colour/predictive.rs` | Decode colour arguments, construct references, reconstruct and convert components |
//! | `src/alpha.rs` | Decode alpha methods, prepare references, apply residuals, retain picture state |
//! | `src/bits.rs`, `src/entropy.rs` | Read bits, runs, mode symbols, layouts, and coefficients |
//! | `src/prediction.rs`, `src/transform.rs` | Calculate predictions and colour residuals |
//! | `src/surface.rs`, `src/error.rs` | Check plane copies; classify failures and attach positions |
//! | `src/bin/spi-decode.rs`, `tests/corpus.rs` | CLI and optional local corpus regression checks |
//!
//! These paths are source locations, not links to private items from the public API. Internal
//! module introductions expand the corresponding stage. For maintenance, follow the data from
//! `format` through `decoder`, then the relevant colour or alpha dispatcher, before changing a
//! shared prediction or entropy helper.
//!
//! # CLI and existing checks
//!
//! The default `cli` feature builds `spi-decode`. For an existing local input, run `cargo run
//! --locked --offline --bin spi-decode -- picture.spi picture.png`. The CLI reads the whole input,
//! checks the header, decodes it, and writes PNG through a temporary file in the output directory.
//! An existing output requires `--force`. Options `--max-input-bytes` and `--max-memory-bytes` use
//! the API defaults. File and decode errors return exit status 1; argument errors use Clap's
//! status 2. PNG encoding is outside the decoder memory account.
//!
//! Run library tests with `cargo test --locked --offline --lib` and examples with `cargo test
//! --locked --offline --doc`. Build public documentation with `cargo doc --locked --offline
//! --no-deps --lib`, and add `--document-private-items` for implementation pages.
//!
//! The optional corpus test uses `tests/local-corpus/manifest.json` by default and skips if that
//! directory is absent. Run `cargo test --locked --offline --test corpus -- --nocapture`. A
//! version-1 manifest contains `cases`; each case has `file`, `input_sha256`, a `header` with
//! width, height, selector when inspection should succeed, and `expect`. Image expectations use
//! `outcome: "image"` and `rgba_sha256`; error expectations use `outcome: "error"` and the stable
//! `kind` string. The RGBA hash covers visible bytes in row order. `SPI_TEST_MANIFEST` can require
//! a different manifest; for work on this repository, use only files inside this repository and
//! clear an external override. Do not search outside the checkout for missing corpus data.
//!
//! # Terminology
//!
//! | Term | Meaning here |
//! | --- | --- |
//! | Pixel / sample / plane | Output position / one component value / an array of one component |
//! | Block / argument / slot | 16 by 16 method unit / component region / square prediction and coefficient region |
//! | Reference / bank | A value used for prediction / an array that retains reference values |
//! | Residual / coefficient | A change added to prediction / a coded value used to reconstruct that change in colour |
//! | Symbol / context | A decoded syntax choice / stored information used to interpret a choice |
//! | Stride / padding | Distance between row starts / storage or coded samples outside visible dimensions |
//! | Record slice | Horizontal picture region assigned to one record |

#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::missing_docs_in_private_items)]
#![warn(
    clippy::bool_to_int_with_if,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_ptr_alignment,
    clippy::cast_sign_loss,
    clippy::char_lit_as_u8,
    clippy::checked_conversions,
    clippy::comparison_chain,
    clippy::dbg_macro,
    clippy::debug_assert_with_mut_call,
    clippy::doc_link_with_quotes,
    clippy::doc_markdown,
    clippy::empty_line_after_outer_attr,
    clippy::float_cmp_const,
    clippy::float_cmp,
    clippy::float_equality_without_abs,
    clippy::manual_assert,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::mod_module_files,
    clippy::option_if_let_else,
    clippy::precedence_bits,
    clippy::redundant_closure_for_method_calls,
    clippy::redundant_else,
    clippy::redundant_pub_crate,
    clippy::semicolon_if_nothing_returned,
    clippy::shadow_unrelated,
    clippy::similar_names,
    clippy::single_char_lifetime_names, // Use 'me over 'a for self lifetimes
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings,
    clippy::todo,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines,
    clippy::unnecessary_cast,
    clippy::unnecessary_literal_bound,
    clippy::unnecessary_wraps,
    clippy::unneeded_field_pattern,
    clippy::unnested_or_patterns,
    clippy::unseparated_literal_suffix,
    clippy::unused_self,
    clippy::used_underscore_binding,
    clippy::useless_let_if_seq,
    clippy::verbose_bit_mask,
    clippy::wildcard_dependencies,
    clippy::wildcard_imports,
    keyword_idents,
    non_ascii_idents,
    noop_method_call,
    unused_crate_dependencies,
    unused_extern_crates,
    unused_import_braces
)]

// These crates are used by the CLI or tests, but the library lint checks all dependencies.
#[cfg(feature = "cli")]
use {clap as _, env_logger as _, tempfile as _};
#[cfg(test)]
use {serde as _, serde_json as _, sha2 as _};

mod alpha;
mod bits;
mod colour;
mod decoder;
mod entropy;
mod error;
mod format;
mod prediction;
mod surface;
mod transform;

pub use error::{
    CoefficientError, DecodeError, ErrorContext, ErrorKind, MotionError, ParseError,
    PredictionError, SyntaxError,
};

/// Describes the visible picture and its coded output form.
///
/// The raw colour-mode and selector values are retained so that callers can inspect a header even
/// when this decoder does not support the encoded form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    /// The visible width in pixels.
    pub width: u32,
    /// The visible height in pixels.
    pub height: u32,
    /// The colour-mode byte from the picture header.
    pub colour_mode: u8,
    /// The output-selector byte from the picture header.
    pub selector: u8,
}

/// Input and accounted-memory limits for one decode call.
///
/// The account includes output, persistent planes, temporary record planes, record views, and
/// context storage. It excludes the caller's input buffer, allocator overhead, and PNG encoding.
/// See the crate page's resource limits section for the Boolean-history accounting discrepancy;
/// this is not an exact heap bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Sets the largest accepted container size in bytes.
    pub max_input_bytes: usize,
    /// Maximum accounted decoder memory in bytes; see the type-level accounting qualification.
    pub max_memory_bytes: usize,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 256 * 1024 * 1024,
            max_memory_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Inspects the framed container and picture header without decoding records.
///
/// This function validates both outer section lengths, rejects trailing container bytes, checks
/// the picture identifier and dimensions, and returns the raw colour mode and output selector. It
/// does not validate the record section or reject an unsupported mode or selector.
///
/// # Errors
///
/// Returns [`DecodeError`] with kind [`ErrorKind::Parse`] when the container framing or picture
/// header is malformed.
pub fn read_header(input: &[u8]) -> Result<ImageInfo, DecodeError> {
    let container = format::parse_container(input)
        .map_err(|error| error::ParseError::Container(Box::new(error)))?;
    let picture = format::parse_picture(container.picture)
        .map_err(|error| error::ParseError::Picture(Box::new(error)))?;
    Ok(ImageInfo {
        width: u32::from(picture.width),
        height: u32::from(picture.height),
        colour_mode: picture.colour_mode,
        selector: picture.selector,
    })
}

/// Decodes one complete picture with default resource limits.
///
/// This is equivalent to calling [`decode_with_limits`] with [`DecodeLimits::default`]. The
/// returned image contains visible pixels only; padding at the right and bottom edges is
/// discarded.
///
/// # Errors
///
/// Returns [`DecodeError`] for malformed or unsupported syntax, incomplete record coverage,
/// resource-limit violations, or allocation failures.
pub fn decode(input: &[u8]) -> Result<image::RgbaImage, DecodeError> {
    decode_with_limits(input, DecodeLimits::default())
}

/// Decodes one complete picture with explicit resource limits.
///
/// The decoder parses the container, validates the supported picture form, reconstructs records in
/// index order, and converts the internal green-blue-red planes plus optional alpha into RGBA
/// pixels. Picture-wide prediction and motion history persists between records.
///
/// A limit is inclusive. Integer overflow during resource accounting is reported as a
/// resource-limit error with a requested size of [`usize::MAX`]. See [`DecodeLimits`] for the
/// memory included in the account.
///
/// # Errors
///
/// Returns [`DecodeError`] for invalid framing or syntax, unsupported picture modes, invalid
/// spatial or motion references, incomplete picture coverage, a breached limit, or an allocation
/// failure.
pub fn decode_with_limits(
    input: &[u8],
    limits: DecodeLimits,
) -> Result<image::RgbaImage, DecodeError> {
    decoder::decode_with_limits(input, limits)
}
