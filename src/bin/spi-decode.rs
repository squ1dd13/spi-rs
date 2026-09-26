//! Decode one SPI image and save its pixels as PNG.
//!
//! The input is read as a complete file. `--max-input-bytes` and
//! `--max-memory-bytes` set the decoder limits. Existing output files require
//! `--force`.

use std::{path::PathBuf, process::ExitCode};

use clap::Parser;
use spi::{DecodeLimits, decode_with_limits, read_header};

#[derive(Debug, Parser)]
#[command(version, about = "Decode an SPI image to PNG")]
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

fn ensure_output_writable(output: &std::path::Path, force: bool) -> Result<(), String> {
    if output.exists() && !force {
        return Err(format!(
            "output file already exists: {} (use --force to replace it)",
            output.display()
        ));
    }
    Ok(())
}

fn run(args: Args) -> Result<(), String> {
    ensure_output_writable(&args.output, args.force)?;

    let input = std::fs::read(&args.input)
        .map_err(|error| format!("cannot read {}: {error}", args.input.display()))?;
    let limits = DecodeLimits {
        max_input_bytes: args.max_input_bytes,
        max_memory_bytes: args.max_memory_bytes,
    };
    let header = read_header(&input).map_err(|error| format!("invalid SPI header: {error}"))?;
    let pixels = decode_with_limits(&input, limits)
        .map_err(|error| format!("cannot decode {}: {error}", args.input.display()))?;
    log::info!(
        "decoded {}x{} image (color mode {}, selector {})",
        header.width,
        header.height,
        header.color_mode,
        header.selector
    );
    pixels
        .save_with_format(&args.output, image::ImageFormat::Png)
        .map_err(|error| format!("cannot write {}: {error}", args.output.display()))
}

fn main() -> ExitCode {
    env_logger::init();
    let args = Args::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("spi-decode: {message}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Args, ensure_output_writable};
    use clap::Parser;

    #[test]
    fn existing_output_requires_force() {
        let existing = std::env::current_exe().expect("test executable path");
        assert!(ensure_output_writable(&existing, false).is_err());
        assert!(ensure_output_writable(&existing, true).is_ok());
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
