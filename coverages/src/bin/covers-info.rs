use std::{fs::File, io::BufReader, path::PathBuf};

use clap::Parser;

use deepsize::DeepSizeOf;
use humansize::{format_size, BINARY};
use log::LevelFilter::Info;
use simple_logger::SimpleLogger;

use coverages::IrreducibleCovers;

#[derive(Parser)]
struct Args {
    /// File to show info for
    infile: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let Args { infile } = Args::parse();
    SimpleLogger::new()
        .without_timestamps()
        .with_level(Info)
        .env()
        .init()
        .expect("Failed to init logger");
    let covers = IrreducibleCovers::bufread(BufReader::new(File::open(&infile)?))?;

    println!("File: {}", infile.to_string_lossy());
    println!("Max size: {}", covers.max_size);
    println!("Total covers: {}", covers.covers.len());
    println!(
        "Unpacked memory: {}",
        format_size(covers.deep_size_of(), BINARY)
    );

    Ok(())
}
