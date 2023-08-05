#![feature(file_create_new)]

use std::{env, fs::File, io::ErrorKind, path::PathBuf};

use anyhow::Context;
use cargo_manifest::Manifest;
use serde::Deserialize;
use simple_logger::SimpleLogger;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    covers_size: u8,
}
impl Default for Metadata {
    fn default() -> Self {
        Self { covers_size: 5 }
    }
}

fn covers(covers_size: u8) -> anyhow::Result<()> {
    let outfile = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join(format!(
        "covers_{}_{}",
        covers_size,
        coverages::VERSION
    ));
    match File::create_new(&outfile) {
        Ok(file) => {
            log::info!("Creating covers");
            let covers = coverages::IrreducibleCovers::compute(covers_size);
            covers.write(file).context("While writing covers")?;
            cargo_emit::rustc_env!("COVERS_FILE", "{}", outfile.display())
        }
        Err(err) if err.kind() == ErrorKind::AlreadyExists => (),
        Err(err) => return Err(err).context("While opening covers file"),
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    SimpleLogger::new()
        .without_timestamps()
        .with_level(log::LevelFilter::Warn)
        .env()
        .init()
        .expect("Failed to init logger");

    let manifest: Manifest<Metadata> =
        Manifest::from_path_with_metadata(PathBuf::from("Cargo.toml"))
            .context("While loading Cargo.toml")?;

    covers(
        manifest
            .package
            .unwrap()
            .metadata
            .unwrap_or_default()
            .covers_size,
    )
    .context("While building static covers")?;

    Ok(())
}
