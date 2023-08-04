#![feature(const_option)]
#![feature(const_try)]
#![feature(const_result_drop)]
#![feature(const_trait_impl)]
#![feature(const_convert)]
#![feature(is_some_and)]

use std::{
    collections::{BTreeSet, HashSet},
    iter::{repeat_with, zip},
    panic::resume_unwind,
    sync::Arc,
};

use async_recursion::async_recursion;
use bitflags::bitflags;
use futures::future::join_all;
use rand::{
    distributions::{Distribution, Uniform, WeightedIndex},
    Rng, SeedableRng,
};
use rand_wyrand::WyRand;
use tokio::sync::{Mutex, OnceCell};

mod rects;
pub use rects::{Rect, Side};

use crate::covers::COVERS;

mod covers;

pub mod config;
pub use config::Config;

/// A maze generator
#[derive(Debug, Clone)]
pub struct Maze {
    /// General config for the maze
    config: Arc<Config>,
    /// root submaze
    root: OnceCell<SubMaze>,
}
impl Maze {
    /// Create a new maze with a given config
    pub fn new(config: Config) -> Self {
        Self {
            config: Arc::new(config),
            root: OnceCell::new(),
        }
    }
    pub async fn draw(&self, rect: Rect, buf: &mut [Tile]) {
        debug_assert_eq!(rect.linearized().map(|l| l.len()), Some(buf.len()));
        self.root
            .get_or_init(|| async {
                SubMaze::new(
                    Rect::MAX,
                    WyRand::seed_from_u64(self.config.seed),
                    self.config.clone(),
                    Default::default(),
                )
            })
            .await
            .draw(rect, &Mutex::new(buf))
            .await
    }
}

/// A maze that covers a given domain
#[derive(Debug, Clone)]
struct SubMaze {
    /// General config for the maze
    config: Arc<Config>,
    /// Domain of this maze
    domain: Rect,
    /// Random number generator seed for this node
    seed: u64,
    /// Content of the submaze
    content: SubMazeContent,
}

#[derive(Debug, Clone)]
enum SubMazeContent {
    Submazes {
        /// Rectangles of the submazes
        rects: Box<[Rect]>,
        /// Doors of the submaze
        doors: Box<[Doors]>,
        /// Submazes
        cells: Box<[OnceCell<SubMaze>]>,
    },
    Room {
        /// Color of the room floor
        color: [u8; 3],
        /// Entrance to the room
        doors: Doors,
    },
}

impl SubMaze {
    fn new(domain: Rect, mut rng: WyRand, config: Arc<Config>, upper_doors: Doors) -> Self {
        debug_assert!(domain.maxx > domain.minx && domain.maxy > domain.miny);
        debug_assert!(upper_doors
            .top
            .iter()
            .all(|x| domain.minx <= *x && *x < domain.maxx));
        debug_assert!(upper_doors
            .bottom
            .iter()
            .all(|x| domain.minx <= *x && *x < domain.maxx));
        debug_assert!(upper_doors
            .left
            .iter()
            .all(|y| domain.miny <= *y && *y < domain.maxy));
        debug_assert!(upper_doors
            .right
            .iter()
            .all(|y| domain.miny <= *y && *y < domain.maxy));
        // First: we need to split further?
        if domain.shape().into_iter().any(|s| s == 1)
            || if config.room_size > 0. {
                domain.linearized().is_some_and(|l| {
                    let p = (-(l.len() as f64 / config.room_size)).exp();
                    rng.gen_bool(p)
                })
            } else {
                false
            }
        {
            log::debug!("{domain:?}: Setting as room.");
            let color = [255; 3];
            // do not split further
            return Self {
                domain,
                config,
                seed: rng.gen(),
                content: SubMazeContent::Room {
                    color,
                    doors: upper_doors,
                },
            };
        }
        // choosing a suitable cover
        let domain_aspect_ratio = domain.aspect_ratio();
        let mut weights = COVERS
            .iter()
            .map(|c| {
                if c.shape[0] as u64 > domain.shape()[0]
                    || c.shape[1] as u64 > domain.shape()[1]
                    || c.shape == [1, 1]
                {
                    // this cover is either too big, or the recursive one
                    return f64::NEG_INFINITY;
                }
                // calculating resulting mean aspect ratio
                let aspect_ratio = c.mean_aspect_ratio() + domain_aspect_ratio;
                // weighting to target 0 (squarish rooms)
                -(aspect_ratio * config.room_squaring_factor).powi(2)
            })
            .collect::<Box<[_]>>();
        let max_weight = *weights
            .iter()
            .max_by(|a, b| {
                a.partial_cmp(b)
                    .expect("All weights should be comparable (not NaN")
            })
            .unwrap();
        for w in weights.iter_mut() {
            *w = (*w - max_weight).exp();
        }
        let idx = WeightedIndex::new(weights.iter()).expect("The max weight should be 1");
        let cover = &COVERS[rng.sample(idx)];
        // choosing the sides of the subcells
        let mut x_sides = sample(&mut rng, domain.minx + 1, domain.maxx, cover.shape[0] - 1);
        x_sides.sort();
        let mut y_sides = sample(&mut rng, domain.miny + 1, domain.maxy, cover.shape[1] - 1);
        y_sides.sort();
        // mapping the rects
        let map_x = |s| {
            if s == 0 {
                domain.minx
            } else if s == cover.shape[0] {
                domain.maxx
            } else {
                x_sides[s as usize - 1]
            }
        };
        let map_y = |s| {
            if s == 0 {
                domain.miny
            } else if s == cover.shape[1] {
                domain.maxy
            } else {
                y_sides[s as usize - 1]
            }
        };
        let rects: Box<[_]> = cover
            .rects
            .into_iter()
            .map(
                |covers::Rect {
                     minx,
                     miny,
                     maxx,
                     maxy,
                 }| Rect {
                    minx: map_x(*minx),
                    miny: map_y(*miny),
                    maxx: map_x(*maxx),
                    maxy: map_y(*maxy),
                },
            )
            .collect();
        #[cfg(debug_assertions)]
        {
            log::debug!("{domain:?}: Checking for subrects collisions");
            for (i, r1) in rects.iter().enumerate() {
                assert!(domain.covers(r1));
                for r2 in rects[..i].iter() {
                    assert!(!r1.collide(r2))
                }
            }
        }
        // Creating doors
        let mut doors: Box<[Doors]> = vec![Default::default(); rects.len()].into_boxed_slice();
        // Choosing a graph submaze
        let graph = maze_graph(&mut rng, &cover.shared_sides);
        for (i, c) in graph.iter().enumerate() {
            // deduplicate to avoid double doors
            for (j, side) in c[..i].iter().enumerate() {
                if let Some(side) = *side {
                    match side {
                        coverages::Side::Vertical { x, miny, maxy } => {
                            let x = map_x(*x);
                            let y = rng.gen_range(map_y(*miny)..map_y(*maxy));
                            let (right_cell, left_cell) = if rects[i].contains(&[x, y]) {
                                (i, j)
                            } else {
                                debug_assert!(rects[j].contains(&[x, y]));
                                (j, i)
                            };
                            doors[right_cell].left.insert(y);
                            doors[left_cell].right.insert(y);
                        }
                        coverages::Side::Orizontal { y, minx, maxx } => {
                            let y = map_y(*y);
                            let x = rng.gen_range(map_x(*minx)..map_x(*maxx));
                            let (bottom_cell, top_cell) = if rects[i].contains(&[x, y]) {
                                (i, j)
                            } else {
                                debug_assert!(rects[j].contains(&[x, y]));
                                (j, i)
                            };
                            doors[bottom_cell].top.insert(x);
                            doors[top_cell].bottom.insert(x);
                        }
                    };
                }
            }
        }
        // Adding the doors from the upper level
        'd: for door_x in upper_doors.top {
            for (r, d) in zip(rects.iter(), doors.iter_mut()) {
                if r.contains(&[door_x, domain.miny]) {
                    d.top.insert(door_x);
                    continue 'd;
                }
            }
            panic!(
                "{domain:?}: No rect found for door [{door_x},{}]",
                domain.miny
            )
        }
        'd: for door_x in upper_doors.bottom {
            for (r, d) in zip(rects.iter(), doors.iter_mut()) {
                if r.contains(&[door_x, domain.maxy - 1]) {
                    d.bottom.insert(door_x);
                    continue 'd;
                }
            }
            panic!(
                "{domain:?}: No rect found for door [{door_x},{}]",
                domain.maxy - 1
            )
        }
        'd: for door_y in upper_doors.left {
            for (r, d) in zip(rects.iter(), doors.iter_mut()) {
                if r.contains(&[domain.minx, door_y]) {
                    d.left.insert(door_y);
                    continue 'd;
                }
            }
            panic!(
                "{domain:?}: No rect found for door [{},{door_y}]",
                domain.minx
            )
        }
        'd: for door_y in upper_doors.right {
            for (r, d) in zip(rects.iter(), doors.iter_mut()) {
                if r.contains(&[domain.maxx - 1, door_y]) {
                    d.right.insert(door_y);
                    continue 'd;
                }
            }
            panic!(
                "{domain:?}: No rect found for door [{},{door_y}]",
                domain.maxx - 1
            )
        }

        log::debug!("{domain:?}: Initializing {} submazes", rects.len());
        Self {
            config,
            domain,
            seed: rng.gen(),
            content: SubMazeContent::Submazes {
                cells: repeat_with(OnceCell::new).take(rects.len()).collect(),
                doors,
                rects,
            },
        }
    }

    #[async_recursion]
    async fn draw(&self, rect: Rect, buf: &Mutex<&mut [Tile]>) {
        match &self.content {
            SubMazeContent::Submazes {
                rects,
                cells,
                doors,
            } => {
                log::debug!("{:?}: Recursing into submazes", self.domain);
                // recurse
                join_all(rects.iter().enumerate().filter_map(|(i, r)| {
                    if r.collide(&rect) {
                        // this room need to be drawn
                        Some(async move {
                            match cells[i]
                                .get_or_try_init(move || {
                                    let r = *r;
                                    let doors = doors[i].clone();
                                    let rng =
                                        WyRand::seed_from_u64(self.seed.wrapping_add(i as u64));
                                    let config = self.config.clone();
                                    // Only the new is spawned in a thread, to avoid messing with the &buf lifetime
                                    tokio::spawn(async move { Self::new(r, rng, config, doors) })
                                })
                                .await
                            {
                                Ok(s) => s.draw(rect, buf).await,
                                Err(err) => match err.try_into_panic() {
                                    Ok(payload) => resume_unwind(payload),
                                    Err(err) => panic!("Submaze generation thread failed: {err}"),
                                },
                            }
                        })
                    } else {
                        None
                    }
                }))
                .await;
            }
            SubMazeContent::Room { color, doors } => {
                log::debug!("{:?}: Drawing room", self.domain);
                let l = rect
                    .linearized()
                    .expect("Cannot draw on a non-linearizable buffer");
                // locking the buffer for the whole drawing step
                let mut buf = buf.lock().await;
                // filling in the color
                for [x, y] in Rect::intersection(&self.domain, &rect).unwrap() {
                    buf[l.global_to_linear(&[x, y])].color = *color;
                }
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
                if let Some(inner) = self
                    .domain
                    .inner()
                    .and_then(|inner| inner.intersection(&rect))
                {
                    for [x, y] in inner {
                        buf[l.global_to_linear(&[x, y])].walls = Walls::empty();
                    }
                }
            }
        }
    }
}

fn sample<R>(rng: &mut R, min: i64, max: i64, amount: u8) -> Vec<i64>
where
    R: Rng + ?Sized,
{
    debug_assert!(max >= min);
    debug_assert!((amount as u64) <= max.abs_diff(min));

    if let Ok(length) = usize::try_from(max.abs_diff(min)) {
        // go with the library one, featuring cool algorithm if amount ~= lenght
        return rand::seq::index::sample(rng, length, amount as _)
            .into_iter()
            .map(|v| min.checked_add_unsigned(v as u64).unwrap())
            .collect();
    }

    // Use rejection, given u8::MAX <<< usize::MAX
    let mut cache = HashSet::with_capacity(amount as usize);
    let distr = Uniform::new(min, max);
    let mut indices = Vec::with_capacity(amount as usize);
    for _ in 0..amount as usize {
        let mut pos = distr.sample(rng);
        while !cache.insert(pos) {
            pos = distr.sample(rng);
        }
        indices.push(pos);
    }

    debug_assert_eq!(indices.len(), amount as usize);
    indices
}

fn maze_graph<'g, R, G, C, T>(rng: &mut R, graph: &'g G) -> Box<[Box<[Option<&'g T>]>]>
where
    R: Rng + ?Sized,
    G: AsRef<[C]>,
    C: AsRef<[Option<T>]> + 'g,
{
    let num_cells = graph.as_ref().len();
    // neighbours function
    // get all neighbours of a given point
    let neighbours = |i: usize| {
        graph.as_ref()[i]
            .as_ref()
            .iter()
            .enumerate()
            .filter_map(move |(j, n)| n.as_ref().map(|p| (p, [i, j])))
    };

    let mut walls = vec![];
    let mut visited = BTreeSet::new();
    let mut maze = vec![vec![None; num_cells].into_boxed_slice(); num_cells].into_boxed_slice();
    // choosing a random starting point
    let start = rng.gen_range(0..num_cells);
    visited.insert(start);
    walls.extend(neighbours(start));

    while !walls.is_empty() {
        let (side, [a, b]) = walls.swap_remove(rng.gen_range(0..walls.len()));
        if visited.contains(&a) != visited.contains(&b) {
            // adding the passage
            maze[a][b] = Some(side);
            maze[b][a] = Some(side);
            // marking as visited and adding the neighbours
            if visited.insert(a) {
                walls.extend(neighbours(a))
            };
            if visited.insert(b) {
                walls.extend(neighbours(b))
            };
        }
    }

    maze
}

bitflags! {
    /// Walls around a tile
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash,Default)]
    pub struct Walls: u8 {
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
pub struct Tile {
    pub color: [u8; 3],
    pub walls: Walls,
}

/// Doors to a section of the maze
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct Doors {
    top: BTreeSet<i64>,
    bottom: BTreeSet<i64>,
    left: BTreeSet<i64>,
    right: BTreeSet<i64>,
}
