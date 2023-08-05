use std::{fs::read_to_string, path::PathBuf, sync::Arc};

use anyhow::{Context};
use clap::Parser;
use serde::Deserialize;
use simple_logger::SimpleLogger;

use maze::{Config as MazeConfig, Doors, Maze, Rect};

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

use bitflags::bitflags;
bitflags! {
    /// Walls around a tile
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash,Default)]
    struct Walls: u8 {
        // Corners
        const TopLeftCorner     = 0b10000000;
        const TopRightCorner    = 0b01000000;
        const BottomLeftCorner  = 0b00100000;
        const BottomRightCorner = 0b00010000;
        // Walls
        const TopWall    = 0b00001000;
        const LeftWall   = 0b00000100;
        const BottomWall = 0b00000010;
        const RightWall  = 0b00000001;
        // Combined walls + corners
        const Top    = Self::TopLeftCorner.bits() | Self::TopWall.bits() | Self::TopRightCorner.bits();
        const Bottom = Self::BottomLeftCorner.bits() | Self::BottomWall.bits() | Self::BottomRightCorner.bits();
        const Left   = Self::TopLeftCorner.bits() | Self::LeftWall.bits() | Self::BottomLeftCorner.bits();
        const Right  = Self::BottomRightCorner.bits() | Self::RightWall.bits() | Self::TopRightCorner.bits();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
struct Tile {
    color: [u8; 3],
    walls: Walls,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(default)]
struct RoomConfig {
    colors: bool,
}

impl Default for RoomConfig {
    fn default() -> Self {
        Self { colors: true }
    }
}

struct Room {
    /// Room rectangle
    domain: Rect,
    /// Color of the room floor
    color: [u8; 3],
    /// Entrance to the room
    doors: Doors,
}
impl maze::Room for Room {
    type Tile = Tile;
    type Config = RoomConfig;

    fn new<R, C>(domain: Rect, doors: Doors, config: &Arc<C>, mut rng: R) -> Self
    where
        Self: Sized,
        R: rand::Rng,
        C: AsRef<RoomConfig>,
    {
        Self {
            domain,
            color: if config.as_ref().as_ref().colors {
                rng.gen()
            } else {
                [255; 3]
            },
            doors,
        }
    }

    fn draw(&self, rect: Rect, buf: &mut [Self::Tile]) {
        let l = rect
            .linearized()
            .expect("Cannot draw on a non-linearizable buffer");

        log::trace!("{:?}: Drawing room.", self.domain);
        // filling in the floor color
        for [x, y] in Rect::intersection(&self.domain, &rect).unwrap() {
            buf[l.global_to_linear(&[x, y])].color = self.color;
        }

        log::trace!("{:?}: Adding walls.", self.domain);
        // top side
        if let Some(side) = self.domain.top().intersection(&rect) {
            for [x, y] in side {
                buf[l.global_to_linear(&[x, y])].walls |= Walls::Top;
            }
        }
        // bottom side
        if let Some(side) = self.domain.bottom().intersection(&rect) {
            for [x, y] in side {
                buf[l.global_to_linear(&[x, y])].walls |= Walls::Bottom;
            }
        }
        // left side
        if let Some(side) = self.domain.left().intersection(&rect) {
            for [x, y] in side {
                buf[l.global_to_linear(&[x, y])].walls |= Walls::Left;
            }
        }
        // right side
        if let Some(side) = self.domain.right().intersection(&rect) {
            for [x, y] in side {
                buf[l.global_to_linear(&[x, y])].walls |= Walls::Right;
            }
        }
        // floor
        // We unset eventual walls, to permit complete redraws
        if let Some(inner) = self
            .domain
            .inner()
            .and_then(|inner| inner.intersection(&rect))
        {
            for [x, y] in inner {
                buf[l.global_to_linear(&[x, y])].walls = Walls::empty();
            }
        }

        log::trace!("{:?}: Adding walls.", self.domain);
        for x in self.doors.top() {
            let p = [*x, self.domain.miny];
            if rect.contains(&p) {
                buf[l.global_to_linear(&p)].walls -= Walls::TopWall;
            }
        }
        for x in self.doors.bottom() {
            let p = [*x, self.domain.maxy - 1];
            if rect.contains(&p) {
                buf[l.global_to_linear(&p)].walls -= Walls::BottomWall;
            }
        }
        for y in self.doors.left() {
            let p = [self.domain.minx, *y];
            if rect.contains(&p) {
                buf[l.global_to_linear(&p)].walls -= Walls::LeftWall;
            }
        }
        for y in self.doors.right() {
            let p = [self.domain.maxx - 1, *y];
            if rect.contains(&p) {
                buf[l.global_to_linear(&p)].walls -= Walls::RightWall;
            }
        }
    }
}
