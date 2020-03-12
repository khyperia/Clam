use crate::{
    input::Input,
    kernel_compilation::RealizedSource,
    parse_vector3,
    setting_value::{SettingValue, SettingValueEnum},
};
use cgmath::{prelude::*, Vector3};
use failure::{err_msg, Error};
use std::{
    fmt::Write as FmtWrite,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Lines, Write},
};

#[derive(Clone, Default, PartialEq)]
pub struct Settings {
    pub values: Vec<SettingValue>,
}

impl Settings {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn get(&self, key: &str) -> Option<&SettingValue> {
        for value in &self.values {
            if value.key() == key {
                return Some(value);
            }
        }
        None
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut SettingValue> {
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

    pub fn write_one(
        &self,
        writer: &mut BufWriter<impl Write>,
        reference: &Settings,
    ) -> Result<(), Error> {
        for value in &self.values {
            if value.key() == "render_scale" {
                continue;
            }
            if let Some(reference) = reference.get(value.key()) {
                if value.value() == reference.value() {
                    continue;
                }
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

    pub fn save(&self, file: &str, realized_source: &RealizedSource) -> Result<(), Error> {
        let file = File::create(file)?;
        let mut writer = BufWriter::new(&file);
        self.write_one(&mut writer, realized_source.default_settings())?;
        Ok(())
    }

    pub fn load_iter<T: BufRead>(
        lines: &mut Lines<T>,
        reference: &Settings,
    ) -> Result<(Settings, bool), Error> {
        let mut result = Settings::new();
        let mut read_any = false;
        for line in lines {
            read_any = true;
            let line = line?;
            if &line == "---" || &line == "" {
                break;
            }
            let split = line.rsplitn(2, '=').collect::<Vec<_>>();
            if split.len() != 2 {
                return Err(err_msg(format!(
                    "Invalid format in settings file: {}",
                    line
                )));
            }
            let key = split[1].trim();
            let new_value = split[0].trim();
            let reference = reference.find(key);
            let val_enum = match *reference.value() {
                SettingValueEnum::Int(_) => SettingValueEnum::Int(new_value.parse()?),
                SettingValueEnum::Float(_, change) => {
                    SettingValueEnum::Float(new_value.parse()?, change)
                }
                SettingValueEnum::Vec3(_, change) => SettingValueEnum::Vec3(
                    parse_vector3(new_value)
                        .ok_or_else(|| failure::err_msg("invalid vector3 in save file"))?,
                    change,
                ),
                SettingValueEnum::Define(_) => SettingValueEnum::Define(new_value.parse()?),
            };
            result
                .values
                .push(SettingValue::new(key.to_string(), val_enum));
        }
        Ok((result, read_any))
    }

    pub fn load(file: &str, reference: &Settings) -> Result<Settings, Error> {
        let file = File::open(file)?;
        let reader = BufReader::new(&file);
        let (loaded, _) = Self::load_iter(&mut reader.lines(), reference)?;
        let mut result = reference.clone();
        result.apply(&loaded);
        Ok(result)
    }

    pub fn status(&self, input: &Input) -> String {
        //let mut keys = self.value_map.keys().collect::<Vec<_>>();
        //keys.sort();
        let mut builder = String::new();
        for (ind, value) in self.values.iter().enumerate() {
            let selected = if ind == input.index { "*" } else { " " };
            let key = value.key();
            match value.value() {
                SettingValueEnum::Int(v) => writeln!(&mut builder, "{} {} = {}", selected, key, v)
                    .expect("Failed to write line to file"),
                SettingValueEnum::Float(v, _) => {
                    writeln!(&mut builder, "{} {} = {}", selected, key, v)
                        .expect("Failed to write line to file")
                }
                SettingValueEnum::Vec3(v, _) => {
                    // TODO
                    writeln!(
                        &mut builder,
                        "{} {} = {} {} {}",
                        selected, key, v.x, v.y, v.z
                    )
                    .expect("Failed to write line to file")
                }
                SettingValueEnum::Define(v) => {
                    writeln!(&mut builder, "{} {} = {}", selected, key, v)
                        .expect("Failed to write line to file")
                }
            }
        }
        builder
    }

    pub fn check_rebuild(&self, against: &Settings) -> bool {
        for value in &self.values {
            if let Some(corresponding) = against.get(value.key()) {
                match (value.value(), corresponding.value()) {
                    (
                        SettingValueEnum::Define(define_value),
                        SettingValueEnum::Define(corresponding),
                    ) => {
                        if define_value != corresponding {
                            return true;
                        }
                    }
                    _ => {
                        if !value.value().kinds_match(corresponding.value()) {
                            return true;
                        }
                    }
                }
            } else {
                return true;
            }
        }
        false
    }

    pub fn normalize(&mut self) {
        let mut look = self.find("look").unwrap_vec3();
        let mut up = self.find("up").unwrap_vec3();
        look = look.normalize();
        up = Vector3::cross(Vector3::cross(look, up), look).normalize();
        *self.find_mut("look").unwrap_vec3_mut() = look;
        *self.find_mut("up").unwrap_vec3_mut() = up;
    }

    pub fn apply(&mut self, other: &Settings) {
        for value in &mut self.values {
            if let Some(other) = other.get(value.key()) {
                if value.value().kinds_match(other.value()) {
                    *value = other.clone();
                }
            }
        }
    }
}
