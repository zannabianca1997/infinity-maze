/// A rectangle
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rect {
    pub minx: i64,
    pub miny: i64,
    pub maxx: i64,
    pub maxy: i64,
}

impl Rect {
    /// Largest possible rectangle
    pub const MAX: Rect = Rect {
        minx: i64::MIN,
        miny: i64::MIN,
        maxx: i64::MAX,
        maxy: i64::MAX,
    };

    /// Check if a points is inside this rect
    /// ```
    /// use maze::Rect;
    ///
    /// let rect = Rect { minx: -3, miny: 4, maxx: 1, maxy: 7};
    /// assert!(rect.contains(&[-3,5]))
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn contains(&self, [x, y]: &[i64; 2]) -> bool {
        self.minx <= *x && *x < self.maxx && self.miny <= *y && *y < self.maxy
    }

    #[inline(always)]
    #[must_use]
    pub const fn covers(&self, other: &Rect) -> bool {
        self.minx <= other.minx
            && other.maxx <= self.maxx
            && self.miny <= other.miny
            && other.maxy <= self.maxy
    }

    #[inline(always)]
    #[must_use]
    pub const fn collide(&self, other: &Rect) -> bool {
        self.maxx > other.minx
            && self.minx < other.maxx
            && self.maxy > other.miny
            && self.miny < other.maxy
    }

    #[inline(always)]
    #[must_use]
    pub const fn shape(&self) -> [u64; 2] {
        [self.maxx.abs_diff(self.minx), self.maxy.abs_diff(self.miny)]
    }

    /// Convert global coordinates to internal ones
    /// ```
    /// use maze::Rect;
    ///
    /// let rect = Rect { minx: -3, miny: 4, maxx: 1, maxy: 7};
    /// assert_eq!(rect.global_to_internal(&[-3,5]), [0,1])
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn global_to_internal(&self, pos: &[i64; 2]) -> [u64; 2] {
        debug_assert!(self.contains(pos));
        [pos[0].abs_diff(self.minx), pos[1].abs_diff(self.miny)]
    }

    /// Convert internal coordinates to global ones
    /// ```
    /// use maze::Rect;
    ///
    /// let rect = Rect { minx: -3, miny: 4, maxx: 1, maxy: 7};
    /// assert_eq!(rect.internal_to_global(&[0,1]), [-3,5])
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn internal_to_global(&self, pos: &[u64; 2]) -> [i64; 2] {
        debug_assert!(pos[0] < self.shape()[0] && pos[1] < self.shape()[1]);
        [
            self.minx.checked_add_unsigned(pos[0]).unwrap(),
            self.miny.checked_add_unsigned(pos[1]).unwrap(),
        ]
    }

    /// If the rectangle is small enough to be represented in memory, get a linearized map
    #[inline(always)]
    #[must_use]
    pub const fn linearized(&self) -> Option<Linearized> {
        let [x, y] = self.shape();
        let x: usize = x.try_into().ok()?;
        let y: usize = y.try_into().ok()?;
        if usize::checked_mul(x, y).is_some() {
            Some(Linearized(self))
        } else {
            None
        }
    }

    /// Logaritm of the aspect ration
    #[inline(always)]
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        (self.maxx.abs_diff(self.minx) as f64).ln() - (self.maxy.abs_diff(self.miny) as f64).ln()
    }

    #[inline(always)]
    #[must_use]
    pub const fn top_left(&self) -> [i64; 2] {
        [self.minx, self.miny]
    }

    /// Find the intersection for two rects
    #[inline(always)]
    #[must_use]
    pub const fn intersection(&self, other: &Rect) -> Option<Rect> {
        if self.collide(other) {
            Some(Rect {
                minx: self.minx.max(other.minx),
                miny: self.miny.max(other.miny),
                maxx: self.maxx.min(other.maxx),
                maxy: self.maxy.min(other.maxy),
            })
        } else {
            None
        }
    }

    /// Top side of the rectangle
    #[inline(always)]
    #[must_use]
    pub const fn top(&self) -> Side {
        let Rect {
            minx, miny, maxx, ..
        } = *self;
        Side::Orizontal {
            y: miny,
            minx,
            maxx,
        }
    }

    /// Bottom side of the rectangle
    #[inline(always)]
    #[must_use]
    pub const fn bottom(&self) -> Side {
        let Rect {
            minx, maxy, maxx, ..
        } = *self;
        Side::Orizontal {
            y: maxy - 1,
            minx,
            maxx,
        }
    }

    /// Bottom side of the rectangle
    #[inline(always)]
    #[must_use]
    pub const fn left(&self) -> Side {
        let Rect {
            minx, maxy, miny, ..
        } = *self;
        Side::Vertical {
            x: minx,
            miny,
            maxy,
        }
    }

    /// Top side of the rectangle
    #[inline(always)]
    #[must_use]
    pub const fn right(&self) -> Side {
        let Rect {
            miny, maxx, maxy, ..
        } = *self;
        Side::Vertical {
            x: maxx - 1,
            miny,
            maxy,
        }
    }

    /// Inner part of the rectangle, if present
    #[inline(always)]
    #[must_use]
    pub const fn inner(&self) -> Option<Rect> {
        if self.shape()[0] > 1 && self.shape()[1] > 1 {
            let Rect {
                miny,
                maxx,
                maxy,
                minx,
            } = *self;
            Some(Rect {
                minx: minx + 1,
                miny: miny + 1,
                maxx: maxx - 1,
                maxy: maxy - 1,
            })
        } else {
            None
        }
    }
}

impl IntoIterator for Rect {
    type Item = [i64; 2];

    type IntoIter = RectIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        if self.shape().into_iter().all(|s| s > 0) {
            RectIntoIter::Running {
                rect: self,
                x: self.minx,
                y: self.miny,
            }
        } else {
            RectIntoIter::Ended
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RectIntoIter {
    Running { rect: Rect, x: i64, y: i64 },
    Ended,
}

impl Iterator for RectIntoIter {
    type Item = [i64; 2];

    fn next(&mut self) -> Option<Self::Item> {
        match *self {
            RectIntoIter::Running {
                rect: Rect { maxx, maxy, .. },
                x,
                y,
            } if x == maxx - 1 && y == maxy - 1 => {
                *self = Self::Ended;
                Some([x, y])
            }
            RectIntoIter::Running {
                rect: Rect { maxx, minx, .. },
                x,
                y,
            } if x == maxx - 1 => {
                let Self::Running { x:nx, y:ny ,..} = self else {unreachable!()};
                *nx = minx;
                *ny += 1;
                Some([x, y])
            }
            RectIntoIter::Running { x, y, .. } => {
                let Self::Running { x:nx ,..} = self else {unreachable!()};
                *nx += 1;
                Some([x, y])
            }
            RectIntoIter::Ended => None,
        }
    }
}

pub struct Linearized<'a>(&'a Rect);

impl Linearized<'_> {
    /// Lenght of the linearized coordinates
    ///
    /// ```
    /// use maze::Rect;
    ///
    /// let rect = Rect { minx: -3, miny: 4, maxx: 1, maxy: 7};
    /// assert_eq!(rect.linearized().unwrap().len(), 12);
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn len(&self) -> usize {
        let [x, y] = self.0.shape();
        x as usize * y as usize
    }

    /// Convert global coordinates to linearized ones
    ///
    /// ```
    /// use maze::Rect;
    ///
    /// let rect = Rect { minx: -3, miny: 4, maxx: 1, maxy: 7};
    /// assert_eq!(rect.linearized().unwrap().global_to_linear(&[-2,5]), 5);
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn global_to_linear(&self, pos: &[i64; 2]) -> usize {
        self.internal_to_linear(&self.0.global_to_internal(pos))
    }

    /// Convert internal coordinates to linearized ones
    ///
    /// ```
    /// use maze::Rect;
    ///
    /// let rect = Rect { minx: -3, miny: 4, maxx: 1, maxy: 7};
    /// assert_eq!(rect.linearized().unwrap().internal_to_linear(&[2,1]), 6);
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn internal_to_linear(&self, pos: &[u64; 2]) -> usize {
        debug_assert!(pos[0] < self.0.shape()[0] && pos[1] < self.0.shape()[1]);
        let stride = self.0.shape()[0] as usize;
        pos[1] as usize * stride + pos[0] as usize
    }

    /// Convert linearized coordinates to global ones
    ///
    /// ```
    /// use maze::Rect;
    ///
    /// let rect = Rect { minx: -3, miny: 4, maxx: 1, maxy: 7};
    /// assert_eq!(rect.linearized().unwrap().linear_to_global(5), [-2,5]);
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn linear_to_global(&self, pos: usize) -> [i64; 2] {
        self.0.internal_to_global(&self.linear_to_internal(pos))
    }

    /// Convert linearized coordinates to internal ones
    ///
    /// ```
    /// use maze::Rect;
    ///
    /// let rect = Rect { minx: -3, miny: 4, maxx: 1, maxy: 7};
    /// assert_eq!(rect.linearized().unwrap().linear_to_internal(6), [1,2]);
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn linear_to_internal(&self, pos: usize) -> [u64; 2] {
        debug_assert!(pos < self.len());
        let stride = self.0.shape()[0] as u64;
        [pos as u64 / stride, pos as u64 % stride]
    }
}

/// An orthogonal line
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Side {
    Vertical { x: i64, miny: i64, maxy: i64 },
    Orizontal { y: i64, minx: i64, maxx: i64 },
}
impl Side {
    #[inline(always)]
    #[must_use]
    pub const fn len(&self) -> i64 {
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

    /// Find the intersection with a rect
    ///
    /// ```
    /// use maze::{Rect,Side};
    ///
    /// let r = Rect {
    ///     minx: -1,
    ///     miny:-2,
    ///     maxx:5,
    ///     maxy:7
    /// };
    /// assert_eq!(
    ///     Side::Vertical { x: 0, miny: 3, maxy: 10 }.intersection(&r),
    ///     Some(Side::Vertical { x: 0, miny: 3, maxy: 7 })
    /// );
    /// assert_eq!(
    ///     Side::Vertical { x: -1, miny: -30, maxy: 6 }.intersection(&r),
    ///     Some(Side::Vertical { x: -1, miny: -2, maxy: 6 })
    /// );
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn intersection(&self, rect: &Rect) -> Option<Side> {
        if self.collide(rect) {
            Some(match *self {
                Side::Vertical { x, miny, maxy } => Side::Vertical {
                    x,
                    miny: miny.max(rect.miny),
                    maxy: maxy.min(rect.maxy),
                },
                Side::Orizontal { y, minx, maxx } => Side::Orizontal {
                    y,
                    minx: minx.max(rect.minx),
                    maxx: maxx.min(rect.maxx),
                },
            })
        } else {
            None
        }
    }

    /// Check if this side collide with a rect
    #[inline(always)]
    #[must_use]
    pub const fn collide(&self, rect: &Rect) -> bool {
        match *self {
            Side::Vertical { x, miny, maxy } => {
                x >= rect.minx && x < rect.maxx && maxy > rect.miny && miny < rect.maxy
            }
            Side::Orizontal { y, minx, maxx } => {
                y >= rect.miny && y < rect.maxy && maxx > rect.minx && minx < rect.maxx
            }
        }
    }
}

impl IntoIterator for Side {
    type Item = [i64; 2];

    type IntoIter = SideIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Side::Vertical { x, miny, maxy } => SideIntoIter::Vertical { x, y: miny, maxy },
            Side::Orizontal { y, minx, maxx } => SideIntoIter::Orizontal { y, x: minx, maxx },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SideIntoIter {
    Vertical { x: i64, y: i64, maxy: i64 },
    Orizontal { y: i64, x: i64, maxx: i64 },
}

impl Iterator for SideIntoIter {
    type Item = [i64; 2];

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SideIntoIter::Vertical { x, y, maxy } => {
                if y < maxy {
                    *y += 1;
                    Some([*x, *y - 1])
                } else {
                    None
                }
            }
            SideIntoIter::Orizontal { y, x, maxx } => {
                if x < maxx {
                    *x += 1;
                    Some([*x - 1, *y])
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    mod rect {
        use super::super::Rect;

        #[test]
        fn into_iter() {
            let r = Rect {
                minx: -1,
                miny: -2,
                maxx: 3,
                maxy: 4,
            };
            let cells: Vec<_> = r.into_iter().collect();
            assert_eq!(
                cells,
                [
                    [-1, -2],
                    [0, -2],
                    [1, -2],
                    [2, -2],
                    [-1, -1],
                    [0, -1],
                    [1, -1],
                    [2, -1],
                    [-1, 0],
                    [0, 0],
                    [1, 0],
                    [2, 0],
                    [-1, 1],
                    [0, 1],
                    [1, 1],
                    [2, 1],
                    [-1, 2],
                    [0, 2],
                    [1, 2],
                    [2, 2],
                    [-1, 3],
                    [0, 3],
                    [1, 3],
                    [2, 3],
                ]
            )
        }
    }

    mod side {
        use super::super::Side;

        #[test]
        fn vertical_into_iter() {
            let s = Side::Vertical {
                x: 42,
                miny: -2,
                maxy: 5,
            };
            let cells: Vec<_> = s.into_iter().collect();
            assert_eq!(
                cells,
                [
                    [42, -2],
                    [42, -1],
                    [42, 0],
                    [42, 1],
                    [42, 2],
                    [42, 3],
                    [42, 4]
                ]
            )
        }
        #[test]
        fn orizontal_into_iter() {
            let s = Side::Orizontal {
                y: 42,
                minx: -2,
                maxx: 5,
            };
            let cells: Vec<_> = s.into_iter().collect();
            assert_eq!(
                cells,
                [
                    [-2, 42],
                    [-1, 42],
                    [0, 42],
                    [1, 42],
                    [2, 42],
                    [3, 42],
                    [4, 42]
                ]
            )
        }
    }
}
