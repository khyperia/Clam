use clam5::*;

pub fn main() -> Result<(), Error> {
    env_logger::builder()
        .filter(Some("clam5"), log::LevelFilter::Trace)
        .init();

    pollster::block_on(run())
}
