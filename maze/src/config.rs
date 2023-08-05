use serde::{Deserialize, Serialize};

/// Config for a maze
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Seed of the maze
    pub seed: u64,
    /// Mean room size
    pub room_size: f64,
    /// Variance of the room aspect ratio
    ///
    /// -> 0   means that oblunged rooms are accepted
    /// -> inf means that rooms are choosen to be the most squarish
    pub squaring_factor: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            room_size: 100.,
            squaring_factor: 4.,
        }
    }
}
