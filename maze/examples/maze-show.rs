use std::{fs::read_to_string, path::PathBuf};

use anyhow::Context;
use clap::Parser;
use serde::Deserialize;
use simple_logger::SimpleLogger;

use maze::simple_rooms::{Room, RoomConfig, Tile, Walls};
use maze::{Config as MazeConfig, Maze, Rect};

#[derive(Debug, Parser)]
struct Args {
    /// Configuration file for the maze
    #[clap(short)]
    config: Option<PathBuf>,
    /// Rectangle to render
    #[clap(flatten)]
    rect: ArgsRect,
    /// Output file
    #[clap(short, long)]
    output: PathBuf,
}

#[derive(Debug, Clone, Copy, Parser)]
struct ArgsRect {
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
impl From<ArgsRect> for Rect {
    fn from(
        ArgsRect {
            minx,
            maxx,
            miny,
            maxy,
        }: ArgsRect,
    ) -> Self {
        Self {
            minx,
            miny,
            maxx,
            maxy,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct Config {
    maze: MazeConfig,
    room: RoomConfig,
}
impl AsRef<MazeConfig> for Config {
    fn as_ref(&self) -> &MazeConfig {
        &self.maze
    }
}
impl AsRef<RoomConfig> for Config {
    fn as_ref(&self) -> &RoomConfig {
        &self.room
    }
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

    let Args {
        config,
        rect,
        output,
    } = Args::parse();
    let rect = Rect::from(rect);
    let config: Config = config
        .map(|path| {
            read_to_string(path)
                .context("Cannot read config file")
                .and_then(|s| toml::from_str(&s).context("Cannot parse config file"))
        })
        .transpose()
        .context("While loading configs")?
        .unwrap_or_default();

    let maze = Maze::<Room, _>::new(config);
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
