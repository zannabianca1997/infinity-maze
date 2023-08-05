use futures::future::join_all;
use rand::{thread_rng, Rng};
use tokio::join;

use maze::simple_rooms::{Room, RoomConfig, Tile};
use maze::{Config as MazeConfig, Maze, Rect};

#[derive(Debug, Default)]
struct Config {
    maze: MazeConfig,
    room: RoomConfig,
}
impl AsRef<MazeConfig> for Config {
    fn as_ref(&self) -> &MazeConfig {
        &self.maze
    }
}
impl AsRef<RoomConfig> for Config {
    fn as_ref(&self) -> &RoomConfig {
        &self.room
    }
}

fn random_config(rng: &mut impl Rng) -> Config {
    Config {
        maze: MazeConfig {
            seed: rng.gen(),
            room_size: rng.gen_range((0.)..(100.)),
            squaring_factor: rng.gen_range((0.)..(10.)),
        },
        room: RoomConfig { colors: true },
    }
}

#[tokio::test]
async fn coherency() {
    join_all((0..10).map(|_| async {
        let maze = Maze::<Room, _>::new(random_config(&mut thread_rng()));
        join_all((0..10).map(|_| async {
            let mut rng = thread_rng();
            // choosing two intersecting rect
            let [x, y] = [(); 2].map(|_| rng.gen_range(i64::MIN + 100..i64::MAX - 100));
            let [r1, r2] = [(); 2].map(|_| Rect {
                minx: rng.gen_range(x - 50..=x),
                maxx: rng.gen_range(x + 1..x + 51),
                miny: rng.gen_range(y - 50..=y),
                maxy: rng.gen_range(y + 1..y + 51),
            });
            debug_assert!(Rect::collide(&r1, &r2));

            // drawing
            let mut t1 = vec![Tile::default(); r1.linearized().unwrap().len()].into_boxed_slice();
            let mut t2 = vec![Tile::default(); r2.linearized().unwrap().len()].into_boxed_slice();
            join!(maze.draw(r1, &mut *t1), maze.draw(r2, &mut *t2));

            for pos in Rect::intersection(&r1, &r2).unwrap() {
                assert_eq!(
                    t1[r1.linearized().unwrap().global_to_linear(&pos)],
                    t2[r2.linearized().unwrap().global_to_linear(&pos)]
                )
            }
        }))
        .await;
    }))
    .await;
}
