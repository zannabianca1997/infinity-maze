use std::{fs::File, io, path::PathBuf};

use clap::Parser;

use coverages::{Coord, IrreducibleCovers};
use log::LevelFilter::Info;
use simple_logger::SimpleLogger;

#[derive(Parser)]
struct Args {
    /// Maximum side to generete
    #[clap(long, short = 'M', default_value = "5")]
    max_size: Coord,
    outfile: PathBuf,
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let Args { max_size, outfile } = Args::parse();
    SimpleLogger::new()
        .without_timestamps()
        .with_level(Info)
        .env()
        .init()
        .expect("Failed to init logger");
    IrreducibleCovers::compute_async(max_size)
        .await
        .write(File::create(outfile)?)
}
