#![feature(const_option)]
#![feature(const_try)]
#![feature(const_result_drop)]
#![feature(const_trait_impl)]
#![feature(const_convert)]
#![feature(is_some_and)]

use std::{collections::HashSet, iter::repeat_with, panic::resume_unwind, sync::Arc};

use async_recursion::async_recursion;
use futures::future::join_all;
use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, Rng, SeedableRng};
use rand_wyrand::WyRand;
use serde::Deserialize;
use tokio::sync::{Mutex, OnceCell};

mod rects;
pub use rects::{Rect, Side};

use crate::covers::COVERS;

mod covers;

/// Config for a maze
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Config {
    /// Seed of the maze
    pub seed: u64,
    /// Mean room size
    pub room_size: f64,
    /// Variance of the room aspect ratio
    /// -> 0   means that oblunged rooms are accepted
    /// -> inf means that rooms are choosen to be the most squarish
    pub room_squaring_factor: f64,
}

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
    async fn draw(&self, rect: Rect, buf: &mut [Tile]) {
        debug_assert_eq!(rect.linearized().map(|l| l.len()), Some(buf.len()));
        self.root
            .get_or_init(|| async {
                SubMaze::new(
                    Rect::MAX,
                    WyRand::seed_from_u64(self.config.seed),
                    self.config.clone(),
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
        /// Submazes
        cells: Box<[OnceCell<SubMaze>]>,
    },
    Room,
}

impl SubMaze {
    fn new(domain: Rect, mut rng: WyRand, config: Arc<Config>) -> Self {
        // First: we need to split further?
        if domain.shape().into_iter().any(|s| s == 1)
            || domain.linearized().is_some_and(|l| {
                let p = (-(l.len() as f64 / config.room_size)).exp();
                rng.gen_bool(p)
            })
        {
            // do not split further
            return Self {
                domain,
                config,
                seed: rng.gen(),
                content: SubMazeContent::Room,
            };
        }
        // choosing a suitable cover
        let aspect_ratio = domain.aspect_ratio();
        let cover = COVERS.choose_weighted(&mut rng, |c| {
            if c.shape[0] as u64 > domain.shape()[0]
                || c.shape[1] as u64 > domain.shape()[1]
                || c.shape == [1, 1]
            {
                // this cover is either too big, or the recursive one
                return 0.;
            }
            // calculating resulting mean aspect ratio
            let aspect_ratio = c.mean_aspect_ratio() + aspect_ratio;
            // weighting to target 0 (squarish rooms)
            (-(aspect_ratio * config.room_squaring_factor).powi(2)).exp()
        }).expect("The weights should be all valids. Maybe `config.room_squaring_factor` was so high it caused everything to be 0?");
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
        Self {
            config,
            domain,
            seed: rng.gen(),
            content: SubMazeContent::Submazes {
                cells: repeat_with(OnceCell::new).take(rects.len()).collect(),
                rects,
            },
        }
    }

    #[async_recursion]
    async fn draw(&self, rect: Rect, buf: &Mutex<&mut [Tile]>) {
        match &self.content {
            SubMazeContent::Submazes { rects, cells } => {
                // recurse
                join_all(rects.iter().enumerate().filter_map(|(i, r)| {
                    if r.collide(&rect) {
                        // this room need to be drawn
                        Some(async move {
                            match cells[i]
                                .get_or_try_init(move || {
                                    let r = *r;
                                    let rng =
                                        WyRand::seed_from_u64(self.seed.wrapping_add(i as u64));
                                    let config = self.config.clone();
                                    // Only the new is spawned in a thread, to avoid messing with the &buf lifetime
                                    tokio::spawn(async move { Self::new(r, rng, config) })
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
            SubMazeContent::Room => {
                let l = rect
                    .linearized()
                    .expect("Cannot draw on a non-linearizable buffer");
                // locking the buffer for the whole drawing step
                let mut buf = buf.lock().await;
                // top-left corner
                if rect.contains(&self.domain.top_left()) {
                    buf[l.global_to_linear(&self.domain.top_left())] = Tile::Both;
                }
                // top side
                let y = self.domain.miny;
                if rect.miny <= y && y < rect.maxy {
                    for x in
                        i64::max(self.domain.minx, rect.minx)..i64::min(self.domain.maxx, rect.maxx)
                    {
                        buf[l.global_to_linear(&[x, y])] = Tile::Up;
                    }
                }
                // left side
                let x = self.domain.minx;
                if rect.minx <= x && x < rect.maxx {
                    for y in
                        i64::max(self.domain.miny, rect.miny)..i64::min(self.domain.maxy, rect.maxy)
                    {
                        buf[l.global_to_linear(&[x, y])] = Tile::Left;
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
    debug_assert!((amount as u64) < max.abs_diff(min));

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Tile {
    None = 0,
    Up = 1,
    Left = 2,
    Both = 3,
}
