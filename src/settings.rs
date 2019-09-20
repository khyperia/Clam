use crate::gl_help::{set_arg_f32, set_arg_u32};
use crate::input::Input;
use crate::setting_value::SettingValue;
use crate::setting_value::SettingValueEnum;
use cgmath::prelude::*;
use cgmath::Vector3;
use failure::err_msg;
use failure::Error;
use gl::types::*;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufRead;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Lines;
use std::io::Write;

#[derive(Clone, Default, PartialEq)]
pub struct Settings {
    pub values: Vec<SettingValue>,
    rebuild: bool,
}

impl Settings {
    pub fn new() -> Self {
        Settings {
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

    pub fn save(&self, file: &str) -> Result<(), Error> {
        let file = File::create(file)?;
        let mut writer = BufWriter::new(&file);
        for value in &self.values {
            match value.value() {
                SettingValueEnum::F32(v, _) => writeln!(&mut writer, "{} = {}", value.key(), v)?,
                SettingValueEnum::U32(v) => writeln!(&mut writer, "{} = {}", value.key(), v)?,
                SettingValueEnum::Define(v) => writeln!(&mut writer, "{} = {}", value.key(), v)?,
            }
        }
        Ok(())
    }

    pub fn save_keyframe(&self, file: &str) -> Result<(), Error> {
        let file = OpenOptions::new().append(true).create(true).open(file)?;
        let mut writer = BufWriter::new(&file);
        for value in &self.values {
            match value.value() {
                SettingValueEnum::F32(v, _) => writeln!(&mut writer, "{} = {}", &value.key(), v)?,
                SettingValueEnum::U32(v) => writeln!(&mut writer, "{} = {}", &value.key(), v)?,
                SettingValueEnum::Define(v) => writeln!(&mut writer, "{} = {}", &value.key(), v)?,
            }
        }
        writeln!(&mut writer, "---")?;
        Ok(())
    }

    pub fn load_iter<T: BufRead>(&mut self, lines: &mut Lines<T>) -> Result<(usize, bool), Error> {
        let mut count = 0;
        for line in lines {
            let line = line?;
            if line == "---" || line == "" {
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
                SettingValueEnum::F32(_, change) => {
                    value.set_value(SettingValueEnum::F32(new_value.parse()?, change));
                }
                SettingValueEnum::U32(_) => {
                    value.set_value(SettingValueEnum::U32(new_value.parse()?));
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
                SettingValueEnum::F32(v, _) => {
                    writeln!(&mut builder, "{}{}{} = {}", selected, constant, key, v)
                        .expect("Failed to write line to file")
                }
                SettingValueEnum::U32(v) => {
                    writeln!(&mut builder, "{}{}{} = {}", selected, constant, key, v)
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
                SettingValueEnum::F32(x, _) => set_arg_f32(compute_shader, value.key(), x)?,
                SettingValueEnum::U32(x) => set_arg_u32(compute_shader, value.key(), x)?,
                SettingValueEnum::Define(_) => (),
            }
        }
        Ok(())
    }

    pub fn normalize(&mut self) {
        let mut look = self.read_vector("look_x", "look_y", "look_z");
        let mut up = self.read_vector("up_x", "up_y", "up_z");
        look = look.normalize();
        up = Vector3::cross(Vector3::cross(look, up), look).normalize();
        self.write_vector(look, "look_x", "look_y", "look_z");
        self.write_vector(up, "up_x", "up_y", "up_z");
    }

    pub fn read_vector(&self, x: &str, y: &str, z: &str) -> Vector3<f32> {
        Vector3::new(
            self.find(x).unwrap_f32(),
            self.find(y).unwrap_f32(),
            self.find(z).unwrap_f32(),
        )
    }

    pub fn write_vector(&mut self, vec: Vector3<f32>, x: &str, y: &str, z: &str) {
        *self.find_mut(x).unwrap_f32_mut() = vec.x;
        *self.find_mut(y).unwrap_f32_mut() = vec.y;
        *self.find_mut(z).unwrap_f32_mut() = vec.z;
    }
}

pub struct KeyframeList {
    base: Settings,
    keyframes: Vec<Settings>,
}

fn interpolate_f32(p0: f32, p1: f32, p2: f32, p3: f32, t: f32, linear: bool) -> f32 {
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

fn interpolate_u32(prev: u32, cur: u32, next: u32, next2: u32, time: f32, linear: bool) -> u32 {
    interpolate_f32(
        prev as f32,
        cur as f32,
        next as f32,
        next2 as f32,
        time,
        linear,
    )
    .round() as u32
}

fn interpolate(
    prev: &SettingValueEnum,
    cur: &SettingValueEnum,
    next: &SettingValueEnum,
    next2: &SettingValueEnum,
    time: f32,
    linear: bool,
) -> SettingValueEnum {
    match (*prev, *cur, *next, *next2) {
        (
            SettingValueEnum::U32(prev),
            SettingValueEnum::U32(cur),
            SettingValueEnum::U32(next),
            SettingValueEnum::U32(next2),
        ) => SettingValueEnum::U32(interpolate_u32(prev, cur, next, next2, time, linear)),
        (
            SettingValueEnum::F32(prev, _),
            SettingValueEnum::F32(cur, delta),
            SettingValueEnum::F32(next, _),
            SettingValueEnum::F32(next2, _),
        ) => SettingValueEnum::F32(interpolate_f32(prev, cur, next, next2, time, linear), delta),
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
    pub fn new(file: &str, mut base: Settings) -> Result<KeyframeList, Error> {
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
        Ok(KeyframeList { base, keyframes })
    }

    // change to isize::mod_euclidian once stablized
    fn mod_euc(lhs: isize, rhs: isize) -> isize {
        let r = lhs % rhs;
        if r < 0 {
            r + rhs.abs()
        } else {
            r
        }
    }

    fn clamp(&self, index: isize, wrap: bool) -> usize {
        let len = self.keyframes.len();
        if wrap {
            Self::mod_euc(index, len as isize) as usize
        } else {
            index.max(0).min(len as isize - 1) as usize
        }
    }

    pub fn interpolate(&mut self, time: f32, wrap: bool) -> &mut Settings {
        let timelen = if wrap {
            self.keyframes.len()
        } else {
            self.keyframes.len() - 1
        };
        let time = time * timelen as f32;
        let index_cur = time as usize;
        let time = time - index_cur as f32;
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
