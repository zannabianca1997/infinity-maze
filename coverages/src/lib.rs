#![feature(iter_collect_into)]

use std::{panic, io};
use std::{collections::{BTreeSet, BTreeMap}, fmt::Display};

use bincode::error::DecodeError;
use bincode::{Encode,Decode};
use flate2::Compression;
use flate2::{read,bufread,write};

pub type Coord = u8;
pub type Area = u16;

/// A rectangle
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Encode,Decode)]
struct Rect {
    minx: Coord,
    miny: Coord,
    maxx: Coord,
    maxy: Coord,
}
impl Rect {
    #[inline(always)]
    #[must_use]
    fn contains(&self, [x, y]: &[Coord; 2]) -> bool {
        self.minx <= *x && *x < self.maxx && self.miny <= *y && *y < self.maxy
    }
    #[inline(always)]
    #[must_use]
    fn covers(&self, other: &Rect) -> bool {
        self.minx <= other.minx
            && other.maxx <= self.maxx
            && self.miny <= other.miny
            && other.maxy <= self.maxy
    }
    #[inline(always)]
    #[must_use]
    fn collide(&self, other: &Rect) -> bool {
        self.maxx > other.minx
            && self.minx < other.maxx
            && self.maxy > other.miny
            && self.miny < other.maxy
    }
    #[inline(always)]
    #[must_use]
    fn area(&self) -> Area {
        (self.maxx - self.minx) as Area * (self.maxy - self.miny) as Area
    }
}

/// A cover of a NxM rectangle
#[derive(Debug, Clone,Encode,Decode,PartialEq, Eq, PartialOrd, Ord)]
pub struct Cover {
    shape: [Coord; 2],
    rects: BTreeSet<Rect>,
}
impl Display for Cover {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stride = self.shape[0] as usize * 2 + 1;
        let mut screen = vec![false; stride * (self.shape[1] as usize * 2 + 1)].into_boxed_slice();
        let rows = screen.chunks_mut(stride).collect::<Box<_>>();
        for r in &self.rects {
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
                    write!(f, " ")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub fn irreducibles(shape: [Coord; 2]) -> BTreeSet<Cover> {
    fn irreducibles(
        shape: [Coord; 2],
        rects: Vec<Rect>,
        x_lines: Vec<Option<Coord>>,
        y_lines: Vec<Option<Coord>>,
    ) -> BTreeSet<Cover> {
        match (0..shape[0])
            .flat_map(|x| (0..shape[1]).map(move |y| [x, y]))
            .filter(|pos| rects.iter().all(|r| !r.contains(pos)))
            .next() // the first is guarantee to be a top-left corner
            {
                // Cover is complete 
                None=> BTreeSet::from([Cover{shape, rects:rects.into_iter().collect()}]),
                // Need recursion
                Some([minx,miny])=> {
                    let mut covers = BTreeSet::new();
                    for maxx in minx+1..=shape[0] {
                        'rects: for maxy in miny+1..=shape[1] {
                            let rect = Rect {minx,maxx,miny,maxy};
                            if rects.iter().all(|r| !r.collide(&rect)) {
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
                            irreducibles(shape, rects, x_lines,y_lines).into_iter().collect_into(&mut covers);
                            }
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


#[derive(Debug, Clone, Encode, Decode)]
pub struct IrreducibleCovers {
    max_size: Coord,
    covers: BTreeMap<[Coord;2], BTreeSet<Cover>>
}
impl IrreducibleCovers {
    #[must_use]
    pub async fn compute_async(max_size:Coord)->Self {
        let mut tasks = Vec::new();
        for w in 1..=max_size {
            for h in 1..=w {
                tasks.push(tokio::spawn( async move { 
                    log::info!("Calculating covers {w}x{h}");
                    let res = ([w,h],irreducibles([w,h]));
                    log::info!("Completed covers {w}x{h}");
                    res
                }));
            }
        }
        let mut covers = BTreeMap::new();
        for task in tasks{
            match task.await {
                Ok((shape, cs)) => covers.insert(shape, cs),
                Err(err) => panic::resume_unwind(err.into_panic()),
            };
        }
        Self { max_size, covers }
    }

    #[must_use]
    pub fn compute(max_size:Coord)->Self {
        let mut covers = BTreeMap::new();
        for w in 1..=max_size {
            for h in 1..=w {
                covers.insert([w,h], irreducibles([w,h]));
                log::info!("Completed covers {w}x{h}");
            }
        }
        Self { max_size, covers }
    } 

    pub fn write(&self,  writer: impl io::Write)-> io::Result<()> {
        let mut writer = write::DeflateEncoder::new(
            writer,
            Compression::best(),
        );
        bincode::encode_into_std_write(self, &mut writer, bincode::config::standard()).map_err(|err| match err {
            bincode::error::EncodeError::Io { inner, .. } => inner,
            other => panic!("IrreducibleCovers should not fail to serialize: {other}"),
        })?;
        writer.finish()?;
        Ok(())
    }
    pub fn read(&self,  reader: impl io::Read)-> Result<Self, DecodeError> {
        let mut reader = read::DeflateDecoder::new(reader);
        bincode::decode_from_std_read(&mut reader, bincode::config::standard())
    }
    pub fn bufread(&self,  reader: impl io::BufRead)-> Result<Self, DecodeError> {
        let mut reader = bufread::DeflateDecoder::new(reader);
        bincode::decode_from_std_read(&mut reader, bincode::config::standard())
    }
}

#[cfg(test)]
mod tests {
    mod all_covers {
        use crate::irreducibles;

        #[test]
        fn one_by_one() {
            assert_eq!(irreducibles([1, 1]).len(), 1)
        }
        #[test]
        fn one_by_two() {
            assert_eq!(irreducibles([1, 2]).len(), 1);
            assert_eq!(irreducibles([2, 1]).len(), 1)
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
