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
        ((self.maxx - self.minx) as f64).ln() - ((self.maxy - self.miny) as f64).ln()
    }

    #[inline(always)]
    #[must_use]
    pub fn top_left(&self) -> [i64; 2] {
        [self.minx, self.maxx]
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
    /// assert_eq!(rect.linearized().unwrap().linear_to_internal(6), [2,1]);
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn linear_to_internal(&self, pos: usize) -> [u64; 2] {
        debug_assert!(pos < self.len());
        let stride = self.0.shape()[0] as u64;
        [pos as u64 % stride, pos as u64 / stride]
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
}
