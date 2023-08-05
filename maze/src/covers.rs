use lazy_static::lazy_static;

pub use coverages::*;

static COVERS_DATA: &[u8] = include_bytes!(env!("COVERS_FILE"));

lazy_static! {
    pub static ref COVERS: Box<[Cover]> = {
        log::info!("Loading static cover data");
        let mut data = COVERS_DATA;
        let covers = IrreducibleCovers::bufread(&mut data)
            .expect("The included covers should be deserializable")
            .covers;
        Vec::from_iter(covers.into_iter()).into_boxed_slice()
    };
}
