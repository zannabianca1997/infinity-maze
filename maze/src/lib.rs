#![feature(const_option)]
#![feature(const_try)]
#![feature(const_result_drop)]
#![feature(const_trait_impl)]
#![feature(const_convert)]
#![feature(is_some_and)]

use std::{
    collections::{HashSet},
    iter::repeat_with,
};

use async_recursion::async_recursion;
use rand::{
    distributions::{Uniform},
    prelude::Distribution,
    seq::{SliceRandom},
    Rng, SeedableRng,
};
use rand_wyrand::WyRand;
use serde::Deserialize;
use tokio::sync::OnceCell;

mod rects;
pub use rects::{Rect, Side};

use crate::covers::COVERS;

mod covers;

/// Config for a maze
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Config {
    /// Mean room size
    pub room_size: f64,
    /// Variance of the room aspect ratio
    /// -> 0   means that oblunged rooms are accepted
    /// -> inf means that rooms are choosen to be the most squarish
    pub room_squaring_factor: f64,
}

#[derive(Debug, Clone)]

struct Doors {
    top: Box<[[i64; 2]]>,
    left: Box<[[i64; 2]]>,
}

/// A maze that covers a given domain
#[derive(Debug, Clone)]
struct SubMaze<'c> {
    /// General config for the maze
    config: &'c Config,
    /// Domain of this maze
    domain: Rect,
    /// Random number generator for this node
    rng: WyRand,
    /// Content of the submaze
    content: SubMazeContent<'c>,
}

#[derive(Debug, Clone)]
enum SubMazeContent<'c> {
    Submazes {
        /// Rectangles of the submazes
        rects: Box<[Rect]>,
        /// Submazes
        cells: Box<[OnceCell<SubMaze<'c>>]>,
    },
    Room,
}

impl<'c> SubMaze<'c> {
    fn new(domain: Rect, mut rng: WyRand, config: &'c Config) -> Self {
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
                rng,
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
            rng,
            content: SubMazeContent::Submazes {
                cells: repeat_with(OnceCell::new).take(rects.len()).collect(),
                rects,
            },
        }
    }

    #[async_recursion]
    async fn draw(&self, rect: Rect, buf: &mut [Tile]) {
        match &self.content {
            SubMazeContent::Submazes { rects, cells } => {
                // recurse
                for (i, r) in rects.iter().enumerate() {
                    if r.collide(&rect) {
                        // this room need to be drawn
                        cells[i]
                            .get_or_init(|| async {
                                let rng = WyRand::seed_from_u64(
                                    self.rng.clone().gen::<u64>().wrapping_add(i as u64),
                                );
                                Self::new(*r, rng, self.config)
                            })
                            .await
                            .draw(rect, buf)
                            .await;
                    }
                }
            }
            SubMazeContent::Room => {
                let l = rect
                    .linearized()
                    .expect("Cannot draw on a non-linearizable buffer");
                debug_assert!(l.len() == buf.len());
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
