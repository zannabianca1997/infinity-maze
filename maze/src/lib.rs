#![feature(const_option)]
#![feature(const_try)]
#![feature(const_result_drop)]
#![feature(const_trait_impl)]
#![feature(const_convert)]

use std::collections::BTreeMap;

use rand_wyrand::WyRand;
use tokio::sync::OnceCell;

mod rects;
pub use rects::{Rect, Side};

mod covers;

#[derive(Debug, Clone)]
struct Doors {
    top: Box<[[i64; 2]]>,
    left: Box<[[i64; 2]]>,
}

/// A maze that covers a given domain
#[derive(Debug, Clone)]
struct SubMaze {
    /// Domain of this maze
    domain: Rect,
    /// Random number generator for this node
    rng: WyRand,
    /// Rectangles of the submazes
    rects: Box<[Rect]>,
    /// Sub sides that need to be opened
    sub_doors: BTreeMap<[usize; 2], Side>,
    /// Opened sides
    doors: Doors,
    /// Submazes
    cells: Box<[OnceCell<SubMaze>]>,
}

impl SubMaze {
    fn new(domain: Rect, rng: WyRand) -> Self {
        todo!()
    }
}
