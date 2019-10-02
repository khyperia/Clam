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

    fn elapsed(&self) -> f64 {
        let now = Instant::now();
        let duration = now - self.start;
        duration.as_secs() as f64 + duration.subsec_nanos() as f64 / 1_000_000_000.0
    }

    pub fn time(&self, value: f64) -> f64 {
        let time = self.elapsed();
        time / value - time
    }

    fn get_min_sec(mut seconds: f64) -> String {
        let minutes = (seconds / 60.0) as u32;
        seconds -= (minutes * 60) as f64;
        if minutes == 0 {
            format!("{:05.2}", seconds)
        } else {
            format!("{:02}:{:02}", minutes, seconds as u32)
        }
    }

    pub fn time_str(&self, value: f64) -> String {
        let left = self.time(value);
        let elapsed = self.elapsed();
        let total = left + elapsed;
        format!(
            "{:05.2}%, {} left, {} elapsed, {} total",
            100.0 * value,
            Self::get_min_sec(left),
            Self::get_min_sec(elapsed),
            Self::get_min_sec(total),
        )
    }
}
