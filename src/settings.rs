use crate::{
    kernel_uniforms::KernelUniforms,
    parse_vector3,
    setting_value::{SettingValue, SettingValueEnum},
    Error,
};
use cgmath::{prelude::*, Vector3};
use std::{
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

    pub fn get_default() -> Self {
        let mut default_settings = Self::new();
        KernelUniforms::fill_defaults(&mut default_settings);
        default_settings.values.push(SettingValue::new(
            "render_scale".to_string(),
            SettingValueEnum::Int(1),
        ));
        default_settings
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
            }
        }
        Ok(())
    }

    pub fn save(&self, file: &str, default_settings: &Settings) -> Result<(), Error> {
        let file = File::create(file)?;
        let mut writer = BufWriter::new(&file);
        self.write_one(&mut writer, default_settings)?;
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
                return Err(format!("Invalid format in settings file: {}", line).into());
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
                    parse_vector3(new_value).ok_or("invalid vector3 in save file")?,
                    change,
                ),
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
