use failure::Error;
use input::Input;
use mandelbox_cfg::DEFAULT_CFG;
use mandelbox_cfg::MandelboxCfg;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write as FmtWrite;
use std::io::BufRead;
use std::io::Write;

#[derive(Copy, Clone)]
pub enum SettingValue {
    U32(u32),
    F32(f32, f32),
}

#[derive(Clone)]
pub struct Settings {
    value_map: HashMap<String, SettingValue>,
    constants: HashSet<String>,
    rebuild: bool,
}

impl Settings {
    pub fn new() -> Self {
        let mut value_map = HashMap::new();
        let mut constants = HashSet::new();
        let mut default = DEFAULT_CFG;
        default.normalize();
        default.write(&mut value_map);
        for (key, _) in &value_map {
            if MandelboxCfg::is_const(key) {
                constants.insert(key.clone());
            }
        }
        Settings {
            value_map,
            constants,
            rebuild: false,
        }
    }

    pub fn constants(&self) -> &HashSet<String> {
        &self.constants
    }

    pub fn clear_constants(&mut self) {
        self.constants.clear()
    }

    pub fn all_constants(&mut self) {
        for (key, _) in &self.value_map {
            self.constants.insert(key.clone());
        }
    }

    pub fn is_const(&self, key: &str) -> bool {
        self.constants.contains(key)
    }

    pub fn set_const(&mut self, key: &str, value: bool) {
        if value {
            self.constants.insert(key.to_string());
        } else {
            self.constants.remove(key);
        }
    }

    pub fn value_map(&self) -> &HashMap<String, SettingValue> {
        &self.value_map
    }

    pub fn insert(&mut self, key: String, value: SettingValue) -> Option<SettingValue> {
        self.value_map.insert(key, value)
    }

    pub fn get(&self, key: &str) -> Option<&SettingValue> {
        self.value_map.get(key)
    }

    pub fn save(&self, file: &str) -> Result<(), Error> {
        let file = ::std::fs::File::create(file)?;
        let mut writer = ::std::io::BufWriter::new(&file);
        for (key, value) in &self.value_map {
            match value {
                &SettingValue::F32(value, _) => writeln!(&mut writer, "{} = {}", key, value)?,
                &SettingValue::U32(value) => writeln!(&mut writer, "{} = {}", key, value)?,
            }
        }
        Ok(())
    }

    pub fn load(&mut self, file: &str) -> Result<(), Error> {
        let file = ::std::fs::File::open(file)?;
        let reader = ::std::io::BufReader::new(&file);
        for line in reader.lines() {
            let line = line?;
            let split = line.rsplitn(2, '=').collect::<Vec<_>>();
            let key = split[1].trim();
            let value = split[0].trim();
            match self.value_map[key] {
                SettingValue::F32(_, change) => self.value_map
                    .insert(key.into(), SettingValue::F32(value.parse()?, change)),
                SettingValue::U32(_) => self.value_map
                    .insert(key.into(), SettingValue::U32(value.parse()?)),
            };
        }
        Ok(())
    }

    pub fn status(&self, input: &Input) -> String {
        let mut keys = self.value_map.keys().collect::<Vec<_>>();
        keys.sort();
        let mut builder = String::new();
        for (ind, key) in keys.iter().enumerate() {
            let is_const = self.is_const(key);
            let selected = if ind == input.index { "*" } else { " " };
            let constant = if is_const { "@" } else { " " };
            match self.value_map[*key] {
                SettingValue::F32(value, _) => {
                    write!(&mut builder, "{}{}{} = {}\n", selected, constant, key, value).unwrap()
                }
                SettingValue::U32(value) => {
                    write!(&mut builder, "{}{}{} = {}\n", selected, constant, key, value).unwrap()
                }
            }
        }
        builder
    }

    pub fn rebuild(&mut self) {
        self.rebuild = true;
    }

    pub fn check_rebuild(&mut self) -> bool {
        let result = self.rebuild;
        self.rebuild = false;
        result
    }
}
