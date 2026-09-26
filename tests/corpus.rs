//! Local corpus regression test.
//!
//! Put a version 1 `manifest.json` and its SPI files in
//! `tests/local-corpus/`. A manifest has `version` and `cases`. Each case has
//! `file`, `input_sha256`, `header` (`width`, `height`, `selector`), and
//! `expect`. The header may be omitted only for an intentionally invalid
//! header. Image expectations use `{"outcome":"image","rgba_sha256":"..."}`;
//! error expectations use `{"outcome":"error","kind":"..."}`. The image
//! hash covers decoded RGBA bytes in row order. By default, the test skips when
//! the default corpus directory is absent. Set `SPI_TEST_MANIFEST` to require
//! another manifest. Input file names are relative to the manifest directory.

use std::{
    collections::HashSet,
    env,
    fmt::Write,
    fs,
    path::{Component, Path, PathBuf},
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use spi::{DecodeLimits, decode_with_limits, read_header};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    file: PathBuf,
    input_sha256: String,
    header: Option<HeaderExpectation>,
    expect: Expectation,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderExpectation {
    width: u32,
    height: u32,
    selector: u8,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
enum Expectation {
    Image { rgba_sha256: String },
    Error { kind: String },
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.version != 1 {
        return Err(format!("unsupported manifest version {}", manifest.version));
    }
    if manifest.cases.is_empty() {
        return Err("manifest has no cases".into());
    }
    let mut files = HashSet::new();
    for case in &manifest.cases {
        if case.file.as_os_str().is_empty()
            || case
                .file
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(format!("unsafe input filename: {}", case.file.display()));
        }
        if !files.insert(&case.file) {
            return Err(format!("duplicate input filename: {}", case.file.display()));
        }
        if !valid_hash(&case.input_sha256) {
            return Err(format!("{} has invalid input_sha256", case.file.display()));
        }
        if let Some(header) = &case.header {
            if header.width == 0 || header.height == 0 {
                return Err(format!(
                    "{} has a zero header dimension",
                    case.file.display()
                ));
            }
        }
        match &case.expect {
            Expectation::Image { .. } if case.header.is_none() => {
                return Err(format!(
                    "{} has no required image header",
                    case.file.display()
                ));
            }
            Expectation::Image { rgba_sha256 } if !valid_hash(rgba_sha256) => {
                return Err(format!("{} has invalid rgba_sha256", case.file.display()));
            }
            Expectation::Error { kind } if kind.is_empty() || kind.trim() != kind => {
                return Err(format!("{} has an invalid error kind", case.file.display()));
            }
            _ => {}
        }
    }
    Ok(())
}

fn resolve_input(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    let canonical_root = root.canonicalize().map_err(|error| {
        format!(
            "cannot resolve corpus directory {}: {error}",
            root.display()
        )
    })?;
    let path = canonical_root.join(relative);
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("cannot resolve {}: {error}", relative.display()))?;
    if !canonical.starts_with(&canonical_root) {
        return Err(format!(
            "input escapes corpus directory: {}",
            relative.display()
        ));
    }
    if !canonical.is_file() {
        return Err(format!("input is not a file: {}", relative.display()));
    }
    Ok(canonical)
}

fn spi_files(root: &Path) -> Result<HashSet<PathBuf>, String> {
    fn walk(root: &Path, current: &Path, files: &mut HashSet<PathBuf>) -> Result<(), String> {
        let entries = fs::read_dir(current).map_err(|error| {
            format!(
                "cannot read corpus directory {}: {error}",
                current.display()
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("cannot read corpus entry: {error}"))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
            if file_type.is_dir() {
                walk(root, &path, files)?;
            } else if file_type.is_file()
                && path.extension().is_some_and(|extension| extension == "spi")
            {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|error| format!("cannot relativize {}: {error}", path.display()))?
                    .to_path_buf();
                files.insert(relative);
            }
        }
        Ok(())
    }

    let mut files = HashSet::new();
    walk(root, root, &mut files)?;
    Ok(files)
}

fn sha256(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut text, "{byte:02x}").expect("writing to String cannot fail");
    }
    text
}

#[test]
fn local_corpus() {
    let override_manifest = env::var_os("SPI_TEST_MANIFEST").map(PathBuf::from);
    let required = env::var("SPI_REQUIRE_CORPUS").is_ok_and(|value| value == "1");
    let manifest_path = override_manifest.clone().unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/local-corpus/manifest.json")
    });
    let default_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/local-corpus");

    if !manifest_path.exists() {
        if override_manifest.is_some() {
            panic!(
                "required corpus manifest is missing: {}",
                manifest_path.display()
            );
        }
        if !default_root.exists() && !required {
            eprintln!(
                "skipping local corpus: {} is absent",
                default_root.display()
            );
            return;
        }
        panic!(
            "required corpus manifest is missing: {}",
            manifest_path.display()
        );
    }

    let manifest_bytes = fs::read(&manifest_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", manifest_path.display()));
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .unwrap_or_else(|error| panic!("invalid manifest {}: {error}", manifest_path.display()));
    validate_manifest(&manifest).unwrap_or_else(|error| panic!("invalid corpus manifest: {error}"));
    let root = manifest_path.parent().expect("manifest path has parent");
    let expected_files: HashSet<PathBuf> = manifest
        .cases
        .iter()
        .map(|case| case.file.clone())
        .collect();
    let present_files =
        spi_files(root).unwrap_or_else(|error| panic!("cannot inspect corpus: {error}"));

    let mut failures = Vec::new();
    for file in present_files.difference(&expected_files) {
        failures.push(format!("{}: unlisted SPI file", file.display()));
    }
    for case in &manifest.cases {
        let result = (|| -> Result<(), String> {
            let path = resolve_input(root, &case.file)?;
            let input = fs::read(&path).map_err(|error| format!("cannot read input: {error}"))?;
            let actual_input_hash = sha256(&input);
            if actual_input_hash != case.input_sha256 {
                return Err(format!("input SHA-256 is {actual_input_hash}"));
            }
            if let Some(expected) = &case.header {
                let header = read_header(&input).map_err(|error| format!("header: {error}"))?;
                if header.width != expected.width || header.height != expected.height {
                    return Err(format!(
                        "header dimensions are {}x{}",
                        header.width, header.height
                    ));
                }
                if header.selector != expected.selector {
                    return Err(format!("header selector is {}", header.selector));
                }
            }
            match &case.expect {
                Expectation::Image { rgba_sha256 } => {
                    let pixels = decode_with_limits(
                        &input,
                        DecodeLimits {
                            max_input_bytes: 256 * 1024 * 1024,
                            max_memory_bytes: 512 * 1024 * 1024,
                        },
                    )
                    .map_err(|error| format!("expected image, decode failed: {error}"))?;
                    let expected = case.header.as_ref().expect("image header validated");
                    if pixels.width() != expected.width || pixels.height() != expected.height {
                        return Err(format!(
                            "decoded dimensions are {}x{}",
                            pixels.width(),
                            pixels.height()
                        ));
                    }
                    let actual = sha256(pixels.as_raw());
                    if actual != *rgba_sha256 {
                        return Err(format!("RGBA SHA-256 is {actual}"));
                    }
                }
                Expectation::Error { kind } => {
                    match decode_with_limits(
                        &input,
                        DecodeLimits {
                            max_input_bytes: 256 * 1024 * 1024,
                            max_memory_bytes: 512 * 1024 * 1024,
                        },
                    ) {
                        Ok(_) => return Err("expected a decode error, got an image".into()),
                        Err(error) if error.kind().to_string() != *kind => {
                            return Err(format!("error kind is {}", error.kind()));
                        }
                        Err(_) => {}
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            failures.push(format!("{}: {error}", case.file.display()));
        }
    }
    assert!(
        failures.is_empty(),
        "corpus failures:\n{}",
        failures.join("\n")
    );
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    fn sample() -> Case {
        Case {
            file: "sample.spi".into(),
            input_sha256: "a".repeat(64),
            header: Some(HeaderExpectation {
                width: 1,
                height: 1,
                selector: 0,
            }),
            expect: Expectation::Image {
                rgba_sha256: "b".repeat(64),
            },
        }
    }

    #[test]
    fn accepts_well_formed_manifest() {
        assert!(
            validate_manifest(&Manifest {
                version: 1,
                cases: vec![sample()]
            })
            .is_ok()
        );
    }

    #[test]
    fn rejects_unsupported_version_duplicate_filename_and_traversal() {
        let good = sample();
        assert!(
            validate_manifest(&Manifest {
                version: 2,
                cases: vec![good.clone()]
            })
            .is_err()
        );
        assert!(
            validate_manifest(&Manifest {
                version: 1,
                cases: vec![good.clone(), good.clone()]
            })
            .is_err()
        );
        let mut escaping = good;
        escaping.file = "../outside.spi".into();
        assert!(
            validate_manifest(&Manifest {
                version: 1,
                cases: vec![escaping]
            })
            .is_err()
        );
    }

    #[test]
    fn rejects_uppercase_hashes() {
        let mut case = sample();
        case.input_sha256 = "A".repeat(64);
        assert!(
            validate_manifest(&Manifest {
                version: 1,
                cases: vec![case]
            })
            .is_err()
        );
    }

    #[test]
    fn missing_listed_input_is_an_error() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
        assert!(resolve_input(&root, Path::new("absent-corpus-case.spi")).is_err());
    }
}
