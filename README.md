# infinity-maze

This is a tool to generate an infinite maze.
To generate an example of what it can do, run
```bash
cargo run --features simple_rooms --example maze-show -- -y 0 -Y 30 -x 0 -X 40 -o test.png
```
and examine the result at `test.png`.

## As a library
Simply define your room type, implementing `maze::Room`, and the config type, implementing `AsRef<Room::Config>` and `AsRef<maze::Config>`, then init the maze with `Maze::new(config)`. `Maze::draw` can be used to fill a buffer of `Room::Tile`