use crate::{
    kernel_compilation::RealizedSource, setting_value::SettingValueEnum, settings::Settings, Error,
};
use cgmath::Vector3;
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

pub struct KeyframeList {
    keyframes: Vec<Settings>,
}

fn interpolate_float(p0: f64, p1: f64, p2: f64, p3: f64, t: f64, linear: bool) -> f64 {
    if linear {
        p1 + (p2 - p1) * t
    } else {
        let t2 = t * t;
        let t3 = t2 * t;
        ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
            / 2.0
    }
}

fn interpolate_vec3(
    p0: Vector3<f64>,
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    t: f64,
    linear: bool,
) -> Vector3<f64> {
    Vector3::new(
        interpolate_float(p0.x, p1.x, p2.x, p3.x, t, linear),
        interpolate_float(p0.y, p1.y, p2.y, p3.y, t, linear),
        interpolate_float(p0.z, p1.z, p2.z, p3.z, t, linear),
    )
}

fn interpolate_int(prev: u64, cur: u64, next: u64, next2: u64, time: f64, linear: bool) -> u64 {
    interpolate_float(
        prev as f64,
        cur as f64,
        next as f64,
        next2 as f64,
        time,
        linear,
    )
    .round() as u64
}

fn interpolate(
    prev: &SettingValueEnum,
    cur: &SettingValueEnum,
    next: &SettingValueEnum,
    next2: &SettingValueEnum,
    time: f64,
    linear: bool,
) -> SettingValueEnum {
    match (prev, cur, next, next2) {
        (
            &SettingValueEnum::Int(prev),
            &SettingValueEnum::Int(cur),
            &SettingValueEnum::Int(next),
            &SettingValueEnum::Int(next2),
        ) => SettingValueEnum::Int(interpolate_int(prev, cur, next, next2, time, linear)),
        (
            &SettingValueEnum::Float(prev, _),
            &SettingValueEnum::Float(cur, delta),
            &SettingValueEnum::Float(next, _),
            &SettingValueEnum::Float(next2, _),
        ) => SettingValueEnum::Float(
            interpolate_float(prev, cur, next, next2, time, linear),
            delta,
        ),
        (
            &SettingValueEnum::Vec3(prev, _),
            &SettingValueEnum::Vec3(cur, delta),
            &SettingValueEnum::Vec3(next, _),
            &SettingValueEnum::Vec3(next2, _),
        ) => SettingValueEnum::Vec3(
            interpolate_vec3(prev, cur, next, next2, time, linear),
            delta,
        ),
        (
            &SettingValueEnum::Define(_),
            &SettingValueEnum::Define(cur),
            &SettingValueEnum::Define(_),
            &SettingValueEnum::Define(_),
        ) => SettingValueEnum::Define(cur),
        _ => panic!("Inconsistent keyframe types"),
    }
}

impl KeyframeList {
    pub fn new() -> KeyframeList {
        KeyframeList {
            keyframes: Vec::new(),
        }
    }

    pub fn load(file: &str, realized_source: &RealizedSource) -> Result<Self, Error> {
        let file = File::open(file)?;
        let reader = BufReader::new(&file);
        let mut lines = reader.lines();
        let mut running_settings = realized_source.default_settings().clone();
        let mut keyframes = Vec::new();
        loop {
            let (new_settings, read_any) = Settings::load_iter(&mut lines, &running_settings)?;
            if !read_any {
                break;
            }
            running_settings.apply(&new_settings);
            keyframes.push(running_settings.clone());
        }
        Ok(Self { keyframes })
    }

    pub fn save(&self, file: &str, realized_source: &RealizedSource) -> Result<(), Error> {
        let file = File::create(file)?;
        let mut writer = BufWriter::new(&file);
        let mut previous = realized_source.default_settings();
        for keyframe in &self.keyframes {
            keyframe.write_one(&mut writer, previous)?;
            writeln!(&mut writer, "---")?;
            previous = keyframe;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    pub fn push(&mut self, keyframe: Settings) {
        self.keyframes.push(keyframe);
    }

    fn clamp(&self, index: isize, wrap: bool) -> usize {
        let len = self.keyframes.len();
        if wrap {
            index.rem_euclid(len as isize) as usize
        } else {
            index.max(0).min(len as isize - 1) as usize
        }
    }

    pub fn interpolate(&self, time: f64, wrap: bool) -> Settings {
        let timelen = if wrap {
            self.keyframes.len()
        } else {
            self.keyframes.len() - 1
        };
        let time = time * timelen as f64;
        let index_cur = time as usize;
        let time = time - index_cur as f64;
        let index_prev = self.clamp(index_cur as isize - 1, wrap);
        let index_next = self.clamp(index_cur as isize + 1, wrap);
        let index_next2 = self.clamp(index_cur as isize + 2, wrap);
        let mut base = self.keyframes[index_cur].clone();
        for value in &mut base.values {
            let prev = self.keyframes[index_prev].find(&value.key()).value();
            let cur = self.keyframes[index_cur].find(&value.key()).value();
            let next = self.keyframes[index_next].find(&value.key()).value();
            let next2 = self.keyframes[index_next2].find(&value.key()).value();
            let result = interpolate(
                prev,
                cur,
                next,
                next2,
                time,
                self.keyframes.len() <= 2 && !wrap,
            );
            value.set_value(result);
        }
        base.normalize();
        base
    }
}
