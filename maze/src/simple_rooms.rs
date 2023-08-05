use std::sync::Arc;

use bitflags::bitflags;
use serde::Deserialize;

use crate::{Doors, Rect};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(default)]
pub struct RoomConfig {
    colors: bool,
}

impl Default for RoomConfig {
    fn default() -> Self {
        Self { colors: true }
    }
}

pub struct Room {
    /// Room rectangle
    domain: Rect,
    /// Color of the room floor
    color: [u8; 3],
    /// Entrance to the room
    doors: Doors,
}
impl crate::Room for Room {
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
