//! Read one complete SPI container and write its visible pixels as PNG.
//!
//! Run `spi-decode INPUT OUTPUT` to decode a file. `--max-input-bytes` and `--max-memory-bytes` use the same defaults as the library. The input reader checks metadata and also limits bytes read, so a growing file cannot bypass the input bound. PNG encoding is outside the decoder memory account.
//!
//! The output is first written to a temporary file in the output directory. Without `--force`, publication fails if the destination exists, including a file created after encoding started. With `--force`, publication replaces it. Decode and I/O errors return status 1; Clap retains status 2 for argument errors. RGB values are preserved under zero alpha.

#![warn(missing_docs, clippy::missing_docs_in_private_items)]

use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::Parser;
use spi::{DecodeLimits, decode_with_limits, read_header};

#[derive(Debug, Parser)]
#[command(version, about = "Decode an SPI image to PNG")]
/// Command-line paths, overwrite policy, and decoder limits.
///
/// Clap derives validation and help text from this structure. Byte limits use the same defaults as [`DecodeLimits::default`].
struct Args {
    /// Input SPI file.
    input: PathBuf,
    /// Output PNG file.
    output: PathBuf,
    /// Replace the output file when it already exists.
    #[arg(short, long)]
    force: bool,
    /// Maximum compressed input size in bytes.
    #[arg(long, default_value_t = 256 * 1024 * 1024)]
    max_input_bytes: usize,
    /// Maximum decoder memory use in bytes.
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    max_memory_bytes: usize,
}

/// Reads a complete SPI file without exceeding the input limit.
///
/// File metadata provides an early size check. The bounded reader still takes at most `max_input_bytes + 1` bytes so growing files and inaccurate metadata cannot bypass the limit. I/O and limit failures are converted to path-aware user messages.
fn read_input(path: &Path, max_input_bytes: usize) -> Result<Vec<u8>, String> {
    let file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let max_len = u64::try_from(max_input_bytes).unwrap_or(u64::MAX);
    let known_len = file
        .metadata()
        .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?
        .len();
    if known_len > max_len {
        return Err(format!(
            "input {} exceeds the {max_input_bytes}-byte limit",
            path.display()
        ));
    }
    let mut input = Vec::new();
    file.take(max_len.saturating_add(1))
        .read_to_end(&mut input)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if input.len() > max_input_bytes {
        return Err(format!(
            "input {} exceeds the {max_input_bytes}-byte limit",
            path.display()
        ));
    }
    Ok(input)
}

/// Publishes a staged PNG at its final path.
///
/// Without `force`, the no-clobber persist operation fails atomically if the target exists. With `force`, the platform's temporary-file persistence replaces the target. Failures become user-facing messages and preserve the temporary file until this function returns.
fn publish_png(staged: tempfile::NamedTempFile, output: &Path, force: bool) -> Result<(), String> {
    let result = if force {
        staged.persist(output)
    } else {
        staged.persist_noclobber(output)
    };
    result.map(|_| ()).map_err(|error| {
        if !force && error.error.kind() == std::io::ErrorKind::AlreadyExists {
            format!(
                "output file already exists: {} (use --force to replace it)",
                output.display()
            )
        } else {
            format!("cannot publish {}: {}", output.display(), error.error)
        }
    })
}

/// Encodes pixels to a staged PNG and publishes it atomically.
///
/// The temporary file is created in the output directory so publication does not cross file systems. Encoding and buffered writes complete before [`publish_png`] applies the requested overwrite policy.
fn save_png(pixels: &image::RgbaImage, output: &Path, force: bool) -> Result<(), String> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| format!("cannot stage {}: {error}", output.display()))?;
    {
        let mut writer = BufWriter::new(staged.as_file_mut());
        pixels
            .write_to(&mut writer, image::ImageFormat::Png)
            .map_err(|error| format!("cannot encode {}: {error}", output.display()))?;
        writer
            .flush()
            .map_err(|error| format!("cannot write {}: {error}", output.display()))?;
    }
    publish_png(staged, output, force)
}

/// Reads, inspects, decodes, and saves one command-line input.
///
/// The same input limit applies during file reading and decoder validation. Header inspection supplies the informational log, while decode and PNG failures receive source or destination path context.
fn run(args: &Args) -> Result<(), String> {
    let input = read_input(&args.input, args.max_input_bytes)?;
    let limits = DecodeLimits {
        max_input_bytes: args.max_input_bytes,
        max_memory_bytes: args.max_memory_bytes,
    };
    let header = read_header(&input).map_err(|error| format!("invalid SPI header: {error}"))?;
    let pixels = decode_with_limits(&input, limits)
        .map_err(|error| format!("cannot decode {}: {error}", args.input.display()))?;
    log::info!(
        "decoded {}x{} image (colour mode {}, selector {})",
        header.width,
        header.height,
        header.colour_mode,
        header.selector
    );
    save_png(&pixels, &args.output, args.force)
}

fn main() -> ExitCode {
    env_logger::init();
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("spi-decode: {message}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Args, publish_png, read_input, save_png};
    use clap::Parser;
    use std::{fs, io::Write};

    #[test]
    fn input_limit_is_checked_while_reading() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("input.spi");
        fs::write(&input, b"abcd").unwrap();
        assert!(read_input(&input, 3).is_err());
        assert_eq!(read_input(&input, 4).unwrap(), b"abcd");
    }

    #[test]
    fn publication_checks_for_an_output_created_after_staging() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("output.png");
        let mut staged = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        staged.write_all(b"new image").unwrap();
        fs::write(&output, b"existing image").unwrap();
        assert!(publish_png(staged, &output, false).is_err());
        assert_eq!(fs::read(output).unwrap(), b"existing image");
    }

    #[test]
    fn save_png_requires_force_to_replace_an_output() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("output.png");
        let pixels = image::RgbaImage::from_raw(1, 1, vec![17, 23, 41, 0]).unwrap();
        fs::write(&output, b"existing image").unwrap();
        assert!(save_png(&pixels, &output, false).is_err());
        assert_eq!(fs::read(&output).unwrap(), b"existing image");
        save_png(&pixels, &output, true).unwrap();
        let decoded = image::open(output).unwrap().into_rgba8();
        assert_eq!(decoded.as_raw(), pixels.as_raw());
    }

    #[test]
    fn argument_errors_keep_clap_status() {
        let error = Args::try_parse_from(["spi-decode", "input.spi"]).unwrap_err();
        assert_eq!(error.exit_code(), 2);
        let args = Args::try_parse_from([
            "spi-decode",
            "input.spi",
            "out.png",
            "--max-input-bytes",
            "42",
            "--force",
        ])
        .unwrap();
        assert_eq!(args.max_input_bytes, 42);
        assert!(args.force);
    }

    #[test]
    fn png_keeps_rgb_under_zero_alpha() {
        let image = image::RgbaImage::from_raw(1, 1, vec![17, 23, 41, 0]).unwrap();
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let decoded =
            image::load_from_memory_with_format(encoded.get_ref(), image::ImageFormat::Png)
                .unwrap()
                .into_rgba8();
        assert_eq!(decoded.as_raw(), &[17, 23, 41, 0]);
    }
}
