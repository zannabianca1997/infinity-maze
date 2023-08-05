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
pub struct Maze<RoomT, ConfigT> {
    /// General config for the maze
    config: Arc<ConfigT>,
    /// root submaze
    root: OnceCell<SubMaze<RoomT, ConfigT>>,
}
impl<RoomT, ConfigT> Maze<RoomT, ConfigT> {
    /// Create a new maze with a given config
    pub fn new(config: ConfigT) -> Self {
        Self {
            config: Arc::new(config),
            root: OnceCell::new(),
        }
    }
}
impl<RoomT, ConfigT> Maze<RoomT, ConfigT>
where
    RoomT: Room,
    ConfigT: AsRef<Config> + AsRef<RoomT::Config>,
{
    pub async fn draw(&self, rect: Rect, buf: &mut [RoomT::Tile]) {
        debug_assert_eq!(rect.linearized().map(|l| l.len()), Some(buf.len()));
        self.root
            .get_or_init(|| async {
                SubMaze::new(
                    Rect::MAX,
                    WyRand::seed_from_u64(<ConfigT as AsRef<Config>>::as_ref(&self.config).seed),
                    &self.config,
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
enum SubMaze<RoomT, ConfigT> {
    Submazes {
        /// Domain of this maze
        domain: Rect,
        /// General config for the maze
        config: Arc<ConfigT>,
        /// Rectangles of the submazes
        rects: Box<[Rect]>,
        /// Doors of the submaze
        doors: Box<[Doors]>,
        /// Submazes
        cells: Box<[OnceCell<SubMaze<RoomT, ConfigT>>]>,
        /// Random number generator seed for this node
        seed: u64,
    },
    Room(RoomT),
}

impl<RoomT, ConfigT> SubMaze<RoomT, ConfigT>
where
    RoomT: Room,
    ConfigT: AsRef<Config> + AsRef<RoomT::Config>,
{
    fn new(domain: Rect, mut rng: WyRand, config: &Arc<ConfigT>, upper_doors: Doors) -> Self {
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
        let maze_config: &Config = config.as_ref().as_ref();
        // First: we need to split further?
        if domain.shape().into_iter().any(|s| s == 1)
            || if maze_config.room_size > 0. {
                domain.linearized().is_some_and(|l| {
                    let p = (-(l.len() as f64 / maze_config.room_size)).exp();
                    rng.gen_bool(p)
                })
            } else {
                false
            }
        {
            log::debug!("{domain:?}: Generating room.");
            // do not split further
            return SubMaze::Room(RoomT::new(domain, upper_doors, &config, rng));
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
                -(aspect_ratio * maze_config.squaring_factor).powi(2)
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

        log::trace!("{domain:?}: Initializing {} submazes", rects.len());
        Self::Submazes {
            domain,
            config: config.clone(),
            seed: rng.gen(),
            cells: repeat_with(OnceCell::new).take(rects.len()).collect(),
            doors,
            rects,
        }
    }

    #[async_recursion(?Send)]
    async fn draw(&self, rect: Rect, buf: &Mutex<&mut [RoomT::Tile]>) {
        match &self {
            SubMaze::Submazes {
                domain,
                config,
                seed,
                rects,
                cells,
                doors,
            } => {
                log::trace!("{:?}: Recursing into submazes", domain);
                // recurse
                join_all(rects.iter().enumerate().filter_map(|(i, r)| {
                    if r.collide(&rect) {
                        // this room need to be drawn
                        Some(async move {
                            cells[i]
                                .get_or_init(move || async move {
                                    Self::new(
                                        *r,
                                        WyRand::seed_from_u64(seed.wrapping_add(i as u64)),
                                        config,
                                        doors[i].clone(),
                                    )
                                })
                                .await
                                .draw(rect, buf)
                                .await
                        })
                    } else {
                        None
                    }
                }))
                .await;
            }
            SubMaze::Room(room) => room.draw(rect, &mut *buf.lock().await),
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

/// Doors to a section of the maze
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Doors {
    top: BTreeSet<i64>,
    bottom: BTreeSet<i64>,
    left: BTreeSet<i64>,
    right: BTreeSet<i64>,
}

impl Doors {
    /// Location of the doors at the top of the room
    pub fn top(&self) -> &BTreeSet<i64> {
        &self.top
    }

    /// Location of the doors at the bottom of the room
    pub fn bottom(&self) -> &BTreeSet<i64> {
        &self.bottom
    }

    /// Location of the doors at the left of the room
    pub fn left(&self) -> &BTreeSet<i64> {
        &self.left
    }

    /// Location of the doors at the right of the room
    pub fn right(&self) -> &BTreeSet<i64> {
        &self.right
    }
}

/// Type of the maze rooms
pub trait Room {
    type Tile;
    type Config;

    fn new<R, C>(domain: Rect, doors: Doors, config: &Arc<C>, rng: R) -> Self
    where
        Self: Sized,
        R: Rng,
        C: AsRef<Self::Config>;

    fn draw(&self, rect: Rect, buf: &mut [Self::Tile]);
}
