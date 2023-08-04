use std::{fs::read_to_string, path::PathBuf};

use anyhow::{bail, Context};
use clap::Parser;
use maze::{config::PartialConfig, Config, Maze, Rect, Tile, Walls};
use simple_logger::SimpleLogger;

#[derive(Debug, Parser)]
struct Args {
    /// Configuration file for the maze
    #[clap(short = 'c', long = "config")]
    file_config: Option<PathBuf>,
    /// Overwrite configuration
    #[clap(flatten)]
    cli_config: PartialConfig,
    /// Rectangle to render
    #[clap(flatten)]
    rect: RenderRect,
    /// Output file
    #[clap(short, long)]
    output: PathBuf,
}

#[derive(Debug, Clone, Copy, Parser)]
struct RenderRect {
    /// Inclusive minimum bound on the x axis
    #[clap(short = 'x', long)]
    minx: i64,
    /// Exclusive maximum bound on the x axis
    #[clap(short = 'X', long)]
    maxx: i64,
    /// Inclusive minimum bound on the y axis
    #[clap(short = 'y', long)]
    miny: i64,
    /// Exclusive maximum bound on the y axis
    #[clap(short = 'Y', long)]
    maxy: i64,
}
impl From<RenderRect> for Rect {
    fn from(
        RenderRect {
            minx,
            maxx,
            miny,
            maxy,
        }: RenderRect,
    ) -> Self {
        Self {
            minx,
            miny,
            maxx,
            maxy,
        }
    }
}

fn parse_args() -> anyhow::Result<(Config, Rect, PathBuf)> {
    let Args {
        file_config,
        cli_config,
        rect,
        output,
    } = Args::parse();
    let config = if let Some(file_config) = file_config {
        toml::from_str::<PartialConfig>(
            &read_to_string(file_config).context("While reading config file")?,
        )
        .context("While parsing config file")?
    } else {
        Default::default()
    }
    .merge(cli_config)
    .or_defaults();
    let rect = Rect::from(rect);
    if rect.linearized().is_none() {
        bail!("The asked rectangle is too big to linearize!")
    }
    Ok((config, rect, output))
}

const TILE_SIZE: u32 = 10;
const WALL_SIZE: u32 = 1;
const WALL_COLOR: [u8; 3] = [0, 0, 0];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    SimpleLogger::new()
        .without_timestamps()
        .with_level(if cfg!(debug_assertions) {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .env()
        .init()
        .context("While initializing logging")?;
    let (config, rect, output) = parse_args().context("While loading configuration")?;
    let maze = Maze::new(config);
    let lin = rect.linearized().unwrap();
    let mut buf = vec![Tile::default(); lin.len()].into_boxed_slice();
    maze.draw(rect, &mut buf).await;
    let mut image = image::RgbImage::new(
        TILE_SIZE * rect.shape()[0] as u32,
        TILE_SIZE * rect.shape()[1] as u32,
    );
    for pos in 0..lin.len() {
        let [top, left] = lin.linear_to_internal(pos).map(|x| x as u32 * TILE_SIZE);
        let tile = buf[pos];
        // top - left
        let color = if tile.walls.contains(Walls::TopLeftCorner) {
            WALL_COLOR
        } else {
            tile.color
        };
        for x in left..left + WALL_SIZE {
            for y in top..top + WALL_SIZE {
                image[(x, y)].0 = color;
            }
        }
        // top
        let color = if tile.walls.contains(Walls::TopWall) {
            WALL_COLOR
        } else {
            tile.color
        };
        for x in left + WALL_SIZE..left + TILE_SIZE - WALL_SIZE {
            for y in top..top + WALL_SIZE {
                image[(x, y)].0 = color;
            }
        }
        // top - right
        let color = if tile.walls.contains(Walls::TopRightCorner) {
            WALL_COLOR
        } else {
            tile.color
        };
        for x in left + TILE_SIZE - WALL_SIZE..left + TILE_SIZE {
            for y in top..top + WALL_SIZE {
                image[(x, y)].0 = color;
            }
        }
        // left
        let color = if tile.walls.contains(Walls::LeftWall) {
            WALL_COLOR
        } else {
            tile.color
        };
        for x in left..left + WALL_SIZE {
            for y in top + WALL_SIZE..top + TILE_SIZE - WALL_SIZE {
                image[(x, y)].0 = color;
            }
        }
        // center
        let color = tile.color;
        for x in left + WALL_SIZE..left + TILE_SIZE - WALL_SIZE {
            for y in top + WALL_SIZE..top + TILE_SIZE - WALL_SIZE {
                image[(x, y)].0 = color;
            }
        }
        // right
        let color = if tile.walls.contains(Walls::RightWall) {
            WALL_COLOR
        } else {
            tile.color
        };
        for x in left + TILE_SIZE - WALL_SIZE..left + TILE_SIZE {
            for y in top + WALL_SIZE..top + TILE_SIZE - WALL_SIZE {
                image[(x, y)].0 = color;
            }
        }
        // bottom - left
        let color = if tile.walls.contains(Walls::BottomLeftCorner) {
            WALL_COLOR
        } else {
            tile.color
        };
        for x in left..left + WALL_SIZE {
            for y in top + TILE_SIZE - WALL_SIZE..top + TILE_SIZE {
                image[(x, y)].0 = color;
            }
        }
        // bottom
        let color = if tile.walls.contains(Walls::BottomWall) {
            WALL_COLOR
        } else {
            tile.color
        };
        for x in left + WALL_SIZE..left + TILE_SIZE - WALL_SIZE {
            for y in top + TILE_SIZE - WALL_SIZE..top + TILE_SIZE {
                image[(x, y)].0 = color;
            }
        }
        // bottom - right
        let color = if tile.walls.contains(Walls::BottomRightCorner) {
            WALL_COLOR
        } else {
            tile.color
        };
        for x in left + TILE_SIZE - WALL_SIZE..left + TILE_SIZE {
            for y in top + TILE_SIZE - WALL_SIZE..top + TILE_SIZE {
                image[(x, y)].0 = color;
            }
        }
    }
    image.save(output).context("While saving the image")?;
    Ok(())
}
