use clam5::*;

pub fn main() {
    env_logger::builder()
        .filter(Some("clam5"), log::LevelFilter::Trace)
        .init();

    run(Box::new(pollster::block_on));
}
