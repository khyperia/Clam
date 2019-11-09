use crate::{
    input::Input,
    parse_vector3,
    setting_value::{SettingValue, SettingValueEnum},
};
use cgmath::{prelude::*, Vector3};
use failure::{err_msg, Error};
use gl::types::*;
use khygl::{set_arg_f32, set_arg_f32_3, set_arg_u32};
use std::{
    fmt::Write as FmtWrite,
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Lines, Write},
};

#[derive(Clone, Default, PartialEq)]
pub struct Settings {
    pub values: Vec<SettingValue>,
    rebuild: bool,
}

impl Settings {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            rebuild: false,
        }
    }

    pub fn clear_constants(&mut self) {
        for value in &mut self.values {
            value.set_const(false);
        }
    }

    pub fn all_constants(&mut self) {
        for value in &mut self.values {
            value.set_const(true);
        }
    }

    fn get(&self, key: &str) -> Option<&SettingValue> {
        for value in &self.values {
            if value.key() == key {
                return Some(value);
            }
        }
        None
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut SettingValue> {
        for value in &mut self.values {
            if value.key() == key {
                return Some(value);
            }
        }
        None
    }

    pub fn find(&self, key: &str) -> &SettingValue {
        match self.get(key) {
            Some(v) => v,
            None => panic!("Key not found: {}", key),
        }
    }

    pub fn find_mut(&mut self, key: &str) -> &mut SettingValue {
        match self.get_mut(key) {
            Some(v) => v,
            None => panic!("Key not found: {}", key),
        }
    }

    fn write_one(&self, writer: &mut BufWriter<impl Write>) -> Result<(), Error> {
        for value in &self.values {
            if value.key() == "render_scale" {
                continue;
            }
            match value.value() {
                SettingValueEnum::Int(v) => writeln!(writer, "{} = {}", value.key(), v)?,
                SettingValueEnum::Float(v, _) => writeln!(writer, "{} = {}", value.key(), v)?,
                SettingValueEnum::Vec3(v, _) => {
                    writeln!(writer, "{} = {} {} {}", value.key(), v.x, v.y, v.z)?
                }
                SettingValueEnum::Define(v) => writeln!(writer, "{} = {}", value.key(), v)?,
            }
        }
        Ok(())
    }

    pub fn save(&self, file: &str) -> Result<(), Error> {
        let file = File::create(file)?;
        let mut writer = BufWriter::new(&file);
        self.write_one(&mut writer)?;
        Ok(())
    }

    pub fn save_keyframe(&self, file: &str) -> Result<(), Error> {
        let file = OpenOptions::new().append(true).create(true).open(file)?;
        let mut writer = BufWriter::new(&file);
        self.write_one(&mut writer)?;
        writeln!(&mut writer, "---")?;
        Ok(())
    }

    pub fn load_iter<T: BufRead>(&mut self, lines: &mut Lines<T>) -> Result<(usize, bool), Error> {
        let mut count = 0;
        for line in lines {
            let line = line?;
            if &line == "---" || &line == "" {
                return Ok((count, true));
            }
            let split = line.rsplitn(2, '=').collect::<Vec<_>>();
            if split.len() != 2 {
                return Err(err_msg(format!(
                    "Invalid format in settings file: {}",
                    line
                )));
            }
            count += 1;
            let key = split[1].trim();
            let new_value = split[0].trim();
            let value = self.find_mut(key);
            match *value.value() {
                SettingValueEnum::Int(_) => {
                    value.set_value(SettingValueEnum::Int(new_value.parse()?));
                }
                SettingValueEnum::Float(_, change) => {
                    value.set_value(SettingValueEnum::Float(new_value.parse()?, change));
                }
                SettingValueEnum::Vec3(_, change) => {
                    value.set_value(SettingValueEnum::Vec3(
                        parse_vector3(new_value)
                            .ok_or_else(|| failure::err_msg("invalid vector3 in save file"))?,
                        change,
                    ));
                }
                SettingValueEnum::Define(_) => {
                    value.set_value(SettingValueEnum::Define(new_value.parse()?));
                }
            };
        }
        Ok((count, false))
    }

    pub fn load(&mut self, file: &str) -> Result<(), Error> {
        let file = File::open(file)?;
        let reader = BufReader::new(&file);
        self.load_iter(&mut reader.lines())?;
        Ok(())
    }

    pub fn status(&self, input: &Input) -> String {
        //let mut keys = self.value_map.keys().collect::<Vec<_>>();
        //keys.sort();
        let mut builder = String::new();
        for (ind, value) in self.values.iter().enumerate() {
            let selected = if ind == input.index { "*" } else { " " };
            let constant = if value.is_const() { "@" } else { " " };
            let key = value.key();
            match value.value() {
                SettingValueEnum::Int(v) => {
                    writeln!(&mut builder, "{}{}{} = {}", selected, constant, key, v)
                        .expect("Failed to write line to file")
                }
                SettingValueEnum::Float(v, _) => {
                    writeln!(&mut builder, "{}{}{} = {}", selected, constant, key, v)
                        .expect("Failed to write line to file")
                }
                SettingValueEnum::Vec3(v, _) => {
                    // TODO
                    writeln!(
                        &mut builder,
                        "{}{}{} = {} {} {}",
                        selected, constant, key, v.x, v.y, v.z
                    )
                    .expect("Failed to write line to file")
                }
                SettingValueEnum::Define(v) => {
                    writeln!(&mut builder, "{}{}{} = {}", selected, constant, key, v)
                        .expect("Failed to write line to file")
                }
            }
        }
        builder
    }

    pub fn rebuild(&mut self) {
        self.rebuild = true;
    }

    pub fn check_rebuild(&mut self) -> bool {
        fn check(v: &mut bool) -> bool {
            let result = *v;
            *v = false;
            result
        }
        let mut result = check(&mut self.rebuild);
        for value in &mut self.values {
            result |= value.check_reset_needs_rebuild();
        }
        result
    }

    pub fn define_variable(&mut self, name: &str, new_value: SettingValueEnum, is_const: bool) {
        let old_value = self.get_mut(name);
        if let Some(old_value) = old_value {
            // TODO: Type change check
            old_value.set_default_value(new_value);
        } else {
            self.values
                .push(SettingValue::new(name.to_string(), new_value, is_const));
        }
    }

    pub fn delete_variable(&mut self, name: &str) {
        self.values.retain(|e| e.key() != name);
    }

    pub fn set_uniforms(&self, compute_shader: GLuint) -> Result<(), Error> {
        for value in &self.values {
            match *value.value() {
                SettingValueEnum::Int(x) => set_arg_u32(compute_shader, value.key(), x as u32)?,
                SettingValueEnum::Float(x, _) => {
                    set_arg_f32(compute_shader, value.key(), x as f32)?
                }
                SettingValueEnum::Vec3(x, _) => set_arg_f32_3(
                    compute_shader,
                    value.key(),
                    x.x as f32,
                    x.y as f32,
                    x.z as f32,
                )?,
                SettingValueEnum::Define(_) => (),
            }
        }
        Ok(())
    }

    pub fn normalize(&mut self) {
        let mut look = self.find("look").unwrap_vec3();
        let mut up = self.find("up").unwrap_vec3();
        look = look.normalize();
        up = Vector3::cross(Vector3::cross(look, up), look).normalize();
        *self.find_mut("look").unwrap_vec3_mut() = look;
        *self.find_mut("up").unwrap_vec3_mut() = up;
    }
}

pub struct KeyframeList {
    base: Settings,
    keyframes: Vec<Settings>,
}

fn interpolate_float(p0: f64, p1: f64, p2: f64, p3: f64, t: f64, linear: bool) -> f64 {
    if linear {
        p1 + (p2 - p1) * t
    } else {
        let t2 = t * t;
        let t3 = t2 * t;
        (((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
            / 2.0)
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
    match (*prev, *cur, *next, *next2) {
        (
            SettingValueEnum::Int(prev),
            SettingValueEnum::Int(cur),
            SettingValueEnum::Int(next),
            SettingValueEnum::Int(next2),
        ) => SettingValueEnum::Int(interpolate_int(prev, cur, next, next2, time, linear)),
        (
            SettingValueEnum::Float(prev, _),
            SettingValueEnum::Float(cur, delta),
            SettingValueEnum::Float(next, _),
            SettingValueEnum::Float(next2, _),
        ) => SettingValueEnum::Float(
            interpolate_float(prev, cur, next, next2, time, linear),
            delta,
        ),
        (
            SettingValueEnum::Vec3(prev, _),
            SettingValueEnum::Vec3(cur, delta),
            SettingValueEnum::Vec3(next, _),
            SettingValueEnum::Vec3(next2, _),
        ) => SettingValueEnum::Vec3(
            interpolate_vec3(prev, cur, next, next2, time, linear),
            delta,
        ),
        (
            SettingValueEnum::Define(_),
            SettingValueEnum::Define(cur),
            SettingValueEnum::Define(_),
            SettingValueEnum::Define(_),
        ) => SettingValueEnum::Define(cur),
        _ => panic!("Inconsistent keyframe types"),
    }
}

impl KeyframeList {
    pub fn new(file: &str, mut base: Settings) -> Result<Self, Error> {
        let file = File::open(file)?;
        let reader = BufReader::new(&file);
        let mut lines = reader.lines();
        let mut keyframes = Vec::new();
        loop {
            let (count, more) = base.load_iter(&mut lines)?;
            if !more {
                break;
            }
            if count == 0 {
                continue;
            }
            keyframes.push(base.clone());
        }
        Ok(Self { base, keyframes })
    }

    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    fn clamp(&self, index: isize, wrap: bool) -> usize {
        let len = self.keyframes.len();
        if wrap {
            index.rem_euclid(len as isize) as usize
        } else {
            index.max(0).min(len as isize - 1) as usize
        }
    }

    pub fn interpolate(&mut self, time: f64, wrap: bool) -> &mut Settings {
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
        for value in &mut self.base.values {
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
        self.base.normalize();
        &mut self.base
    }
}
