use std::time::Instant;

pub struct Progress {
    start: Instant,
}

impl Progress {
    pub fn new() -> Progress {
        Progress {
            start: Instant::now(),
        }
    }

    pub fn time(&self, value: f32) -> f32 {
        let now = Instant::now();
        // value / time = 1 / (result + time)
        // time / value - time = result
        let duration = now - self.start;
        let time = duration.as_secs() as f32 + duration.subsec_nanos() as f32 / 1_000_000_000.0;
        time / value - time
    }

    pub fn time_str(&self, value: f32) -> String {
        let mut seconds = self.time(value);
        let minutes = (seconds / 60.0) as u32;
        seconds -= (minutes * 60) as f32;
        format!(
            "{:05.2}%, {:02}:{:05.2} left",
            100.0 * value,
            minutes,
            seconds
        )
    }
}
