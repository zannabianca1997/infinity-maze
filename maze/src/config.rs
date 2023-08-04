use clap::Parser;
use serde::{Deserialize, Serialize};

/// Config for a maze
#[derive(Debug, Clone, Copy, Serialize)]
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

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            room_size: 100.,
            room_squaring_factor: 4.,
        }
    }
}

/// Partial config for a maze
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Parser, Default)]
pub struct PartialConfig {
    /// Seed of the maze
    #[clap(long)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Mean room size
    #[clap(long)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub room_size: Option<f64>,
    /// Variance of the room aspect ratio
    /// -> 0   means that oblunged rooms are accepted
    /// -> inf means that rooms are choosen to be the most squarish
    #[clap(long)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub room_squaring_factor: Option<f64>,
}
impl PartialConfig {
    pub fn merge(self, other: PartialConfig) -> Self {
        Self {
            seed: other.seed.or(self.seed),
            room_size: other.room_size.or(self.room_size),
            room_squaring_factor: other.room_squaring_factor.or(self.room_squaring_factor),
        }
    }
    pub fn or_defaults(self) -> Config {
        let default = Config::default();
        Config {
            seed: self.seed.unwrap_or(default.seed),
            room_size: self.room_size.unwrap_or(default.room_size),
            room_squaring_factor: self
                .room_squaring_factor
                .unwrap_or(default.room_squaring_factor),
        }
    }
}
