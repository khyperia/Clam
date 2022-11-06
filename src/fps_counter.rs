use instant::Instant;

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

        let weight = self.weight / self.spf;
        self.spf = (duration.as_secs_f64() + (self.spf * weight)) / (weight + 1.0);
    }

    pub fn value(&self) -> f64 {
        1.0 / self.spf
    }
}
