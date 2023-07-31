use lazy_static::lazy_static;

pub use coverages::*;

static COVERS_DATA: &[u8] = include_bytes!("../../covers.bin");

lazy_static! {
    pub static ref COVERS: IrreducibleCovers = {
        log::info!("Loading static cover data");
        let mut data = COVERS_DATA;
        IrreducibleCovers::bufread(&mut data).expect("The included covers should be deserializable")
    };
}
