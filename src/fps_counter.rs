use std::time::Instant;

pub struct FpsCounter {
    weight: f64,
    last_fps: Instant,
    spf: f64,
}

impl FpsCounter {
    pub fn new(weight: f64) -> Self {
        Self {
            weight,
            last_fps: Instant::now(),
            spf: 1.0,
        }
    }

    pub fn tick(&mut self) {
        let now = Instant::now();
        let duration = now.duration_since(self.last_fps);
        self.last_fps = now;

        // as_secs returns u64, subsec_nanos returns u32
        let time = duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / 1_000_000_000.0;

        let weight = self.weight / self.spf;
        self.spf = (time + (self.spf * weight)) / (weight + 1.0);
    }

    pub fn value(&self) -> f64 {
        1.0 / self.spf
    }
}
