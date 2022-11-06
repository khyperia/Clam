use clam5::*;

pub fn main() -> Result<(), Error> {
    pollster::block_on(run())
}
