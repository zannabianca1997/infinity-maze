#![feature(iter_collect_into)]

use std::collections::BTreeSet;
use std::fmt::Display;
use std::{io, panic};

use bincode::error::DecodeError;
use bincode::{Decode, Encode};
use deepsize::DeepSizeOf;
use flate2::Compression;
use flate2::{bufread, read, write};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub type Coord = u8;
pub type Area = u16;

/// A rectangle
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Encode, Decode, DeepSizeOf)]
pub struct Rect {
    pub minx: Coord,
    pub miny: Coord,
    pub maxx: Coord,
    pub maxy: Coord,
}
impl Rect {
    #[inline(always)]
    #[must_use]
    const fn contains(&self, [x, y]: &[Coord; 2]) -> bool {
        self.minx <= *x && *x < self.maxx && self.miny <= *y && *y < self.maxy
    }
    #[inline(always)]
    #[must_use]
    const fn covers(&self, other: &Rect) -> bool {
        self.minx <= other.minx
            && other.maxx <= self.maxx
            && self.miny <= other.miny
            && other.maxy <= self.maxy
    }
    #[inline(always)]
    #[must_use]
    const fn collide(&self, other: &Rect) -> bool {
        self.maxx > other.minx
            && self.minx < other.maxx
            && self.maxy > other.miny
            && self.miny < other.maxy
    }
    #[inline(always)]
    #[must_use]
    const fn area(&self) -> Area {
        (self.maxx - self.minx) as Area * (self.maxy - self.miny) as Area
    }

    #[inline(always)]
    #[must_use]
    fn transpose(self) -> Rect {
        let Rect {
            minx,
            miny,
            maxx,
            maxy,
        } = self;
        Rect {
            minx: miny,
            miny: minx,
            maxx: maxy,
            maxy: maxx,
        }
    }

    /// Logaritm of the aspect ration
    #[inline(always)]
    #[must_use]
    fn aspect_ratio(&self) -> f64 {
        ((self.maxx - self.minx) as f64).ln() - ((self.maxy - self.miny) as f64).ln()
    }
}

/// An orthogonal line
#[derive(Debug, Clone, Copy, Encode, Decode, PartialEq, Eq, PartialOrd, Ord, DeepSizeOf)]
pub enum Side {
    Vertical { x: Coord, miny: Coord, maxy: Coord },
    Orizontal { y: Coord, minx: Coord, maxx: Coord },
}
impl Side {
    #[inline(always)]
    #[must_use]
    pub const fn len(&self) -> Coord {
        let (Side::Vertical {
            miny: min,
            maxy: max,
            ..
        }
        | Side::Orizontal {
            minx: min,
            maxx: max,
            ..
        }) = self;
        *max - *min
    }

    #[inline(always)]
    #[must_use]
    pub const fn transpose(self) -> Side {
        match self {
            Side::Vertical { x, miny, maxy } => Side::Orizontal {
                y: x,
                minx: miny,
                maxx: maxy,
            },
            Side::Orizontal { y, minx, maxx } => Side::Vertical {
                x: y,
                miny: minx,
                maxy: maxx,
            },
        }
    }
}

fn shared_sides(rects: &[Rect]) -> Box<[Box<[Option<Side>]>]> {
    let mut graph =
        vec![vec![None; rects.len()].into_boxed_slice(); rects.len()].into_boxed_slice();

    for i in 0..rects.len() {
        for j in 0..i {
            let r1 = &rects[i];
            let r2 = &rects[j];
            let s = if r1.maxx == r2.minx && r1.miny < r2.maxy && r2.miny < r1.maxy {
                Some(Side::Vertical {
                    x: r1.maxx,
                    miny: Coord::max(r1.miny, r2.miny),
                    maxy: Coord::min(r1.maxy, r2.maxy),
                })
            } else if r2.maxx == r1.minx && r1.miny < r2.maxy && r2.miny < r1.maxy {
                Some(Side::Vertical {
                    x: r2.maxx,
                    miny: Coord::max(r1.miny, r2.miny),
                    maxy: Coord::min(r1.maxy, r2.maxy),
                })
            } else if r1.maxy == r2.miny && r1.minx < r2.maxx && r2.minx < r1.maxx {
                Some(Side::Orizontal {
                    y: r1.maxy,
                    minx: Coord::max(r1.minx, r2.minx),
                    maxx: Coord::min(r1.maxx, r2.maxx),
                })
            } else if r2.maxy == r1.miny && r1.minx < r2.maxx && r2.minx < r1.maxx {
                Some(Side::Orizontal {
                    y: r2.maxy,
                    minx: Coord::max(r1.minx, r2.minx),
                    maxx: Coord::min(r1.maxx, r2.maxx),
                })
            } else {
                None
            };
            graph[i][j] = s;
            graph[j][i] = s;
        }
    }

    graph
}

/// A cover of a NxM rectangle
#[derive(Debug, Clone, Encode, Decode, PartialEq, Eq, PartialOrd, Ord, DeepSizeOf)]
pub struct Cover {
    pub shape: [Coord; 2],
    pub rects: Box<[Rect]>,
    pub shared_sides: Box<[Box<[Option<Side>]>]>,
}
impl Cover {
    fn transposed(self) -> Cover {
        let Cover {
            shape: [w, h],
            rects,
            shared_sides,
        } = self;
        Cover {
            shape: [h, w],
            rects: rects.into_vec().into_iter().map(Rect::transpose).collect(),
            shared_sides: shared_sides
                .into_vec()
                .into_iter()
                .map(|r| {
                    r.into_vec()
                        .into_iter()
                        .map(|s| s.map(Side::transpose))
                        .collect()
                })
                .collect(),
        }
    }
    /// Mean aspect ratio of the subrects, if a square is covered
    pub fn mean_aspect_ratio(&self) -> f64 {
        (self.rects.into_iter().map(Rect::aspect_ratio).sum::<f64>() / self.rects.len() as f64)
            - ((self.shape[0] as f64).ln() - (self.shape[1] as f64).ln())
    }
}
impl Display for Cover {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stride = self.shape[0] as usize * 2 + 1;
        let mut screen = vec![false; stride * (self.shape[1] as usize * 2 + 1)].into_boxed_slice();
        let rows = screen.chunks_mut(stride).collect::<Box<_>>();
        for r in self.rects.iter() {
            for x in (2 * r.minx as usize)..(2 * r.maxx as usize + 1) {
                rows[2 * r.miny as usize][x] = true;
                rows[2 * r.maxy as usize][x] = true;
            }
            for y in (2 * r.miny as usize + 1)..(2 * r.maxy as usize) {
                rows[y][2 * r.minx as usize] = true;
                rows[y][2 * r.maxx as usize] = true;
            }
        }
        for r in rows.into_vec() {
            for ch in (&*r).into_iter() {
                if *ch {
                    write!(f, "#")?;
                } else {
                    write!(f, ".")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub fn irreducibles(shape: [Coord; 2]) -> Vec<Cover> {
    fn irreducibles(
        shape: [Coord; 2],
        rects: Vec<Rect>,
        x_lines: Vec<Option<Coord>>,
        y_lines: Vec<Option<Coord>>,
    ) -> Vec<Cover> {
        match (0..shape[0])
            .flat_map(|x| (0..shape[1]).map(move |y| [x, y]))
            .filter(|pos| rects.iter().all(|r| !r.contains(pos)))
            .next() // the first is guarantee to be a top-left corner
            {
                // Cover is complete 
                None=> vec![Cover{shape, shared_sides:shared_sides(&rects),rects:rects.into_boxed_slice()}],
                // Need recursion
                Some([minx,miny])=> {
                    let mut covers = vec![];
                    for maxx in minx+1..=shape[0] {
                        'rects: for maxy in miny+1..=shape[1] {
                            let rect = Rect {minx,maxx,miny,maxy};
                            if rects.iter().any(|r| r.collide(&rect)) {
                                continue 'rects;
                            }

                            // updating lines, checking we did not fill up a line without using it
                            let mut x_lines = x_lines.clone();
                            for y in rect.miny..(rect.maxy-1){
                                if let Some(l) = &mut x_lines[y as usize] {
                                    *l -= rect.maxx-rect.minx;
                                    if *l == 0 {
                                        continue 'rects; // try next rectangle
                                    }
                                }
                            }
                            if rect.miny>1 {
                                x_lines[rect.miny as usize -1]=None
                            }
                            if let Some(l) = x_lines.get_mut(rect.maxy as usize-1) {
                                *l=None
                            }
                            let mut y_lines = y_lines.clone();
                            for x in rect.minx..(rect.maxx-1){
                                if let Some(l) = &mut y_lines[x as usize] {
                                    *l -= rect.maxy-rect.miny;
                                    if *l == 0 {
                                        continue 'rects; // try next rectangle
                                    }
                                }
                            }
                            if rect.minx > 1 {
                                y_lines[rect.minx as usize - 1] = None
                            }
                            if let Some(l) = y_lines.get_mut(rect.maxx as usize-1) {
                                *l = None
                            }
                            // adding the rectangle and checking for irreducibility
                            let mut rects = rects.clone();
                            rects.push(rect);
                            for (i,r1) in rects.iter().enumerate() {    // for all possible couples
                                'c2: for r2 in rects[i+1..].iter() {           // of different rectangles
                                    for (r1,r2) in [(r1,r2),(r2,r1)] {  // in both order
                                        // if they could form the top-lef and bottom-right angles of a rectangle
                                        // that's not the maximum one
                                        if r2.maxx >= r1.maxx
                                            && r2.maxy >= r1.maxy
                                            && r1.miny<=r2.miny
                                            && r1.minx<=r2.minx
                                            && (
                                                // at least one side must detach from the shape side
                                                r1.minx > 0
                                                || r1.miny > 0
                                                || r2.maxx < shape[0]
                                                || r2.maxy < shape[1]
                                            ) {
                                            // Rectangle to test
                                            let suspicius_rect = Rect {
                                                minx: r1.minx,
                                                maxx: r2.maxx,
                                                miny: r1.miny,
                                                maxy: r2.maxy
                                            };
                                            // if the new rect could complete this one
                                            if suspicius_rect.covers(&rect) {
                                                let mut area = suspicius_rect.area();
                                                for r3 in rects.iter() {
                                                    match (suspicius_rect.covers(r3),suspicius_rect.collide(r3)) {
                                                        (true, true) => {
                                                            area-=r3.area();
                                                            if area == 0 {
                                                                // the suspicius rect is completly filled with rectangles that are covered by it.
                                                                // This mean this partial cover is necessarely reducible
                                                                continue 'rects;
                                                            }
                                                        },
                                                        (true, false) => unreachable!(),
                                                        (false, true) => continue 'c2,
                                                        (false, false) => (),
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // recursing
                            covers.append(&mut irreducibles(shape, rects, x_lines,y_lines));
                        }
                    }
                    covers
                }
            }
    }
    irreducibles(
        shape,
        vec![],
        vec![Some(shape[0]); shape[1] as usize - 1],
        vec![Some(shape[1]); shape[0] as usize - 1],
    )
}

#[derive(Debug, Clone, Encode, Decode, DeepSizeOf)]
pub struct IrreducibleCovers {
    pub max_size: Coord,
    pub covers: BTreeSet<Cover>,
}
impl IrreducibleCovers {
    #[must_use]
    pub async fn compute_async(max_size: Coord) -> Self {
        let mut tasks = Vec::new();
        for w in 1..=max_size {
            for h in 1..=w {
                tasks.push(tokio::spawn(async move {
                    log::info!("Calculating covers {w}x{h}");
                    let res = irreducibles([w, h]);
                    log::info!("Completed covers {w}x{h}");
                    res
                }));
            }
        }
        let mut covers = Vec::new();
        for task in tasks {
            match task.await {
                Ok(mut res) => covers.append(&mut res),
                Err(err) => panic::resume_unwind(err.into_panic()),
            };
        }
        Self {
            max_size,
            covers: covers
                .into_iter()
                .flat_map(|c| [c.clone(), c.transposed()])
                .collect(),
        }
    }

    #[must_use]
    pub fn compute(max_size: Coord) -> Self {
        let mut covers = Vec::new();
        for w in 1..=max_size {
            for h in 1..=w {
                covers.append(&mut irreducibles([w, h]));
                log::info!("Completed covers {w}x{h}");
            }
        }
        Self {
            max_size,
            covers: covers
                .into_iter()
                .flat_map(|c| [c.clone(), c.transposed()])
                .collect(),
        }
    }

    pub fn write(&self, writer: impl io::Write) -> io::Result<()> {
        let mut writer = write::DeflateEncoder::new(writer, Compression::best());
        bincode::encode_into_std_write(self, &mut writer, bincode::config::standard()).map_err(
            |err| match err {
                bincode::error::EncodeError::Io { inner, .. } => inner,
                other => panic!("IrreducibleCovers should not fail to serialize: {other}"),
            },
        )?;
        writer.finish()?;
        Ok(())
    }
    pub fn read(reader: impl io::Read) -> Result<Self, DecodeError> {
        let mut reader = read::DeflateDecoder::new(reader);
        bincode::decode_from_std_read(&mut reader, bincode::config::standard())
    }
    pub fn bufread(reader: impl io::BufRead) -> Result<Self, DecodeError> {
        let mut reader = bufread::DeflateDecoder::new(reader);
        bincode::decode_from_std_read(&mut reader, bincode::config::standard())
    }
}

#[cfg(test)]
mod tests {
    mod generation {

        mod graph {
            use crate::{irreducibles, Cover, Side};

            #[test]
            fn one_by_one() {
                let covers = irreducibles([1, 1]);
                let [Cover{shared_sides,..}] = &covers[..] else {unreachable!()};
                assert_eq!(shared_sides.len(), 1);
                assert_eq!(shared_sides[0].len(), 1);
                assert_eq!(shared_sides[0][0], None);
            }

            #[test]
            fn one_by_two() {
                let covers = irreducibles([1, 2]);
                let [Cover{shared_sides,..}] = &covers[..] else {unreachable!()};
                assert_eq!(shared_sides.len(), 2);
                assert_eq!(shared_sides[0].len(), 2);
                assert_eq!(shared_sides[0][0], shared_sides[1][1]);
                assert_eq!(shared_sides[0][0], None);
                assert_eq!(shared_sides[0][1], shared_sides[1][0]);
                assert_eq!(
                    shared_sides[0][1],
                    Some(Side::Orizontal {
                        y: 1,
                        minx: 0,
                        maxx: 1
                    })
                );
            }

            #[test]
            fn two_by_one() {
                let covers = irreducibles([2, 1]);
                let [Cover{shared_sides,..}] = &covers[..] else {unreachable!()};
                assert_eq!(shared_sides.len(), 2);
                assert_eq!(shared_sides[0].len(), 2);
                assert_eq!(shared_sides[0][0], shared_sides[1][1]);
                assert_eq!(shared_sides[0][0], None);
                assert_eq!(shared_sides[0][1], shared_sides[1][0]);
                assert_eq!(
                    shared_sides[0][1],
                    Some(Side::Vertical {
                        x: 1,
                        miny: 0,
                        maxy: 1
                    })
                );
            }

            #[test]
            fn three_by_three() {
                let covers = irreducibles([3, 3]);
                let [Cover{shared_sides:ss1,..},Cover{shared_sides:ss2,..}] = &covers[..] else {unreachable!()};
                for shared_sides in [ss1, ss2] {
                    assert_eq!(shared_sides.len(), 5);
                    for i in 0..5 {
                        assert_eq!(shared_sides[i].len(), 5);
                        for j in 0..5 {
                            assert_eq!(shared_sides[i][j], shared_sides[j][i])
                        }
                    }

                    let sides = shared_sides
                        .into_iter()
                        .flat_map(|l| l.into_iter())
                        .filter_map(|x| *x)
                        .collect::<Vec<_>>();
                    assert_eq!(sides.len(), 2 * 8);
                    assert!(sides.iter().all(|s| s.len() == 1))
                }
            }
        }
        mod counts {
            use crate::irreducibles;

            #[test]
            fn one_by_two() {
                assert_eq!(irreducibles([1, 2]).len(), 1);
                assert_eq!(irreducibles([2, 1]).len(), 1);
            }

            #[test]
            fn two_by_two() {
                assert_eq!(irreducibles([2, 2]).len(), 0)
            }

            #[test]
            fn two_by_three() {
                assert_eq!(irreducibles([2, 3]).len(), 0);
                assert_eq!(irreducibles([3, 2]).len(), 0)
            }

            #[test]
            fn three_by_three() {
                assert_eq!(irreducibles([3, 3]).len(), 2);
                assert_eq!(irreducibles([3, 3]).len(), 2)
            }
        }
    }
}
