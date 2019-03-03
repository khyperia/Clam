use byteorder::{NativeEndian, WriteBytesExt};
use failure::Error;
use input::Input;
use input::Vector;
use regex::Regex;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write as FmtWrite;
use std::io::BufRead;
use std::io::Lines;
use std::io::Write;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SettingValue {
    U32(u32),
    F32(f32, f32),
}

#[derive(Clone, Default, PartialEq)]
pub struct Settings {
    ordered_names: Vec<String>,
    default_values: HashMap<String, SettingValue>,
    value_map: HashMap<String, SettingValue>,
    constants: HashSet<String>,
    rebuild: bool,
}

impl Settings {
    pub fn new() -> Self {
        Settings {
            ordered_names: Vec::new(),
            default_values: HashMap::new(),
            value_map: HashMap::new(),
            constants: HashSet::new(),
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
        for key in &self.ordered_names {
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
            match *value {
                SettingValue::F32(value, _) => writeln!(&mut writer, "{} = {}", key, value)?,
                SettingValue::U32(value) => writeln!(&mut writer, "{} = {}", key, value)?,
            }
        }
        Ok(())
    }

    pub fn save_keyframe(&self, file: &str) -> Result<(), Error> {
        let file = ::std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(file)?;
        let mut writer = ::std::io::BufWriter::new(&file);
        for (key, value) in &self.value_map {
            match *value {
                SettingValue::F32(value, _) => writeln!(&mut writer, "{} = {}", key, value)?,
                SettingValue::U32(value) => writeln!(&mut writer, "{} = {}", key, value)?,
            }
        }
        writeln!(&mut writer, "---")?;
        Ok(())
    }

    pub fn load_iter<T: BufRead>(&mut self, lines: &mut Lines<T>) -> Result<(usize, bool), Error> {
        let mut count = 0;
        for line in lines {
            let line = line?;
            let split = line.rsplitn(2, '=').collect::<Vec<_>>();
            if split.len() != 2 {
                return Ok((count, true));
            }
            count += 1;
            let key = split[1].trim();
            let value = split[0].trim();
            match self.value_map[key] {
                SettingValue::F32(_, change) => self
                    .value_map
                    .insert(key.into(), SettingValue::F32(value.parse()?, change)),
                SettingValue::U32(_) => self
                    .value_map
                    .insert(key.into(), SettingValue::U32(value.parse()?)),
            };
        }
        Ok((count, false))
    }

    pub fn load(&mut self, file: &str) -> Result<(), Error> {
        let file = ::std::fs::File::open(file)?;
        let reader = ::std::io::BufReader::new(&file);
        self.load_iter(&mut reader.lines())?;
        Ok(())
    }

    pub fn nth(&self, index: usize) -> String {
        self.ordered_names[index].clone()
    }

    pub fn default_for(&self, key: &str) -> Option<SettingValue> {
        self.default_values.get(key).cloned()
    }

    pub fn status(&self, input: &Input) -> String {
        //let mut keys = self.value_map.keys().collect::<Vec<_>>();
        //keys.sort();
        let mut builder = String::new();
        for (ind, key) in self.ordered_names.iter().enumerate() {
            let is_const = self.is_const(key);
            let selected = if ind == input.index { "*" } else { " " };
            let constant = if is_const { "@" } else { " " };
            match self.value_map[key] {
                SettingValue::F32(value, _) => {
                    writeln!(&mut builder, "{}{}{} = {}", selected, constant, key, value).unwrap()
                }
                SettingValue::U32(value) => {
                    writeln!(&mut builder, "{}{}{} = {}", selected, constant, key, value).unwrap()
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

    pub fn set_src(&mut self, src: &str) {
        lazy_static! {
            static ref RE: Regex = Regex::new(
               r#"(?m)^ *(?P<kind>float|int) _(?P<name>[a-zA-Z0-9_]+); *// (?P<value>[-+]?\d+(?:\.\d+)?) *(?P<change>[-+]?\d+(?:\.\d+)?)? *(?P<const>const)? *$"#).unwrap();
        }
        self.ordered_names.clear();
        self.value_map.clear();
        let mut once = false;
        for cap in RE.captures_iter(src) {
            once = true;
            println!("match: {}", &cap[0]);
            let kind = &cap["kind"];
            let name = &cap["name"];
            let setting = match kind {
                "float" => {
                    let value = cap["value"].parse().unwrap();
                    let change = cap["change"].parse().unwrap();
                    SettingValue::F32(value, change)
                }
                "int" => {
                    let value = cap["value"].parse().unwrap();
                    SettingValue::U32(value)
                }
                _ => {
                    panic!("Regex returned invalid kind");
                }
            };
            self.ordered_names.push(name.to_string());
            self.value_map.insert(name.to_string(), setting);
            let is_const = cap.name("const").is_some();
            if is_const {
                self.constants.insert(name.to_string());
            } else {
                self.constants.remove(name);
            }
        }
        self.ordered_names.push("render_scale".to_string());
        self.value_map
            .insert("render_scale".to_string(), SettingValue::U32(1));
        assert!(once, "Regex should get at least one setting");
        self.default_values = self.value_map.clone();
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::new();
        for name in &self.ordered_names {
            match self.get(name).expect("Missing setting") {
                &SettingValue::F32(x, _) => result.write_f32::<NativeEndian>(x).unwrap(),
                &SettingValue::U32(x) => result.write_u32::<NativeEndian>(x).unwrap(),
            }
        }
        result
    }

    pub fn normalize(&mut self) {
        fn get_f32(settings: &mut Settings, key: &str) -> (f32, f32) {
            match settings.get(key) {
                Some(&SettingValue::F32(x, scale)) => (x, scale),
                _ => panic!("Missing key {}", key),
            }
        }
        let (look_x, look_x_scale) = get_f32(self, "look_x");
        let (look_y, look_y_scale) = get_f32(self, "look_y");
        let (look_z, look_z_scale) = get_f32(self, "look_z");
        let (up_x, up_x_scale) = get_f32(self, "up_x");
        let (up_y, up_y_scale) = get_f32(self, "up_y");
        let (up_z, up_z_scale) = get_f32(self, "up_z");
        let mut look = Vector::new(look_x, look_y, look_z);
        let mut up = Vector::new(up_x, up_y, up_z);
        look = look.normalized();
        up = Vector::cross(Vector::cross(look, up), look).normalized();
        self.insert(
            "look_x".to_string(),
            SettingValue::F32(look.x, look_x_scale),
        );
        self.insert(
            "look_y".to_string(),
            SettingValue::F32(look.y, look_y_scale),
        );
        self.insert(
            "look_z".to_string(),
            SettingValue::F32(look.z, look_z_scale),
        );
        self.insert("up_x".to_string(), SettingValue::F32(up.x, up_x_scale));
        self.insert("up_y".to_string(), SettingValue::F32(up.y, up_y_scale));
        self.insert("up_z".to_string(), SettingValue::F32(up.z, up_z_scale));
    }
}

pub struct KeyframeList {
    base: Settings,
    keyframes: Vec<Settings>,
}

// template<typename T, typename Tscalar>
// T CatmullRom(T p0, T p1, T p2, T p3, Tscalar t)
// {
//     Tscalar t2 = t * t;
//     Tscalar t3 = t2 * t;
//     return (T)((((Tscalar)2 * p1) + (-p0 + p2) * t +
//         ((Tscalar)2 * p0 - (Tscalar)5 * p1 + (Tscalar)4 * p2 - p3) * t2 +
//         (-p0 + (Tscalar)3 * p1 - (Tscalar)3 * p2 + p3) * t3) / (Tscalar)2);
// }

fn interpolate_f32(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    (((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
        / 2.0)
}

fn interpolate_u32(prev: u32, cur: u32, next: u32, next2: u32, time: f32) -> u32 {
    interpolate_f32(prev as f32, cur as f32, next as f32, next2 as f32, time) as u32
}

fn interpolate(
    prev: SettingValue,
    cur: SettingValue,
    next: SettingValue,
    next2: SettingValue,
    time: f32,
) -> SettingValue {
    match (prev, cur, next, next2) {
        (
            SettingValue::U32(prev),
            SettingValue::U32(cur),
            SettingValue::U32(next),
            SettingValue::U32(next2),
        ) => SettingValue::U32(interpolate_u32(prev, cur, next, next2, time)),
        (
            SettingValue::F32(prev, _),
            SettingValue::F32(cur, delta),
            SettingValue::F32(next, _),
            SettingValue::F32(next2, _),
        ) => SettingValue::F32(interpolate_f32(prev, cur, next, next2, time), delta),
        _ => panic!("Inconsistent keyframe types"),
    }
}

impl KeyframeList {
    pub fn new(file: &str, mut base: Settings) -> Result<KeyframeList, Error> {
        let file = ::std::fs::File::open(file)?;
        let reader = ::std::io::BufReader::new(&file);
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

    pub fn interpolate(&mut self, time: f32) -> &Settings {
        let time = time * (self.keyframes.len() - 1) as f32;
        let index_cur = time as usize;
        let time = time - index_cur as f32;
        let index_prev = if index_cur == 0 { 0 } else { index_cur - 1 };
        let index_next = (index_cur + 1).min(self.keyframes.len() - 1);
        let index_next2 = (index_cur + 2).min(self.keyframes.len() - 1);
        let keys = self.base.value_map.keys().cloned().collect::<Vec<String>>();
        for key in keys {
            let prev = *self.keyframes[index_prev].get(&key).unwrap();
            let cur = *self.keyframes[index_cur].get(&key).unwrap();
            let next = *self.keyframes[index_next].get(&key).unwrap();
            let next2 = *self.keyframes[index_next2].get(&key).unwrap();
            let result = interpolate(prev, cur, next, next2, time);
            self.base.insert(key, result);
        }
        self.base.normalize();
        &self.base
    }
}
