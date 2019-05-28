use byteorder::{NativeEndian, WriteBytesExt};
use failure::err_msg;
use failure::Error;
use input::Input;
use input::Vector;
use regex::Regex;
use std::fmt::Write as FmtWrite;
use std::io::BufRead;
use std::io::Lines;
use std::io::Write;

#[derive(Debug, PartialEq, Clone)]
pub struct SettingValue {
    pub key: String,
    pub value: SettingValueEnum,
    default_value: SettingValueEnum,
    is_const: bool,
    const_changed: bool,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SettingValueEnum {
    U32(u32),
    F32(f32, f32),
    Define(bool),
}

impl SettingValue {
    pub fn change_one(&mut self, increase: bool) {
        match self.value {
            SettingValueEnum::F32(_, _) => (),
            SettingValueEnum::U32(ref mut value) => {
                if increase {
                    *value = *value + 1;
                } else {
                    if *value != 0 {
                        *value = *value - 1;
                    }
                }
            }
            SettingValueEnum::Define(ref mut value) => {
                *value = !*value;
                self.const_changed = true;
            }
        }
    }

    pub fn change(&mut self, increase: bool, mut dt: f32) {
        dt *= if increase { 1.0 } else { -1.0 };
        if let SettingValueEnum::F32(ref mut value, change) = self.value {
            if change < 0.0 {
                *value = *value * (-change + 1.0).powf(dt);
            } else {
                *value = *value + dt * change;
            }
        };
    }

    pub fn toggle(&mut self) {
        fn def(val: &SettingValueEnum) -> Option<bool> {
            match val {
                &SettingValueEnum::Define(v) => Some(v),
                _ => None,
            }
        }
        let old_const = def(&self.value);
        if self.value == self.default_value {
            match self.value {
                SettingValueEnum::U32(ref mut v) => *v = 0,
                SettingValueEnum::F32(ref mut v, _) => *v = 0.0,
                SettingValueEnum::Define(ref mut v) => *v = false,
            }
        } else {
            self.value = self.default_value;
        }
        let new_const = def(&self.value);
        if old_const != new_const {
            self.const_changed = true;
        }
    }

    pub fn toggle_const(&mut self) {
        match self.value {
            SettingValueEnum::Define(_) => (),
            _ => {
                self.is_const = !self.is_const;
                self.const_changed = true;
            }
        }
    }

    pub fn set_const(&mut self, value: bool) {
        match self.value {
            SettingValueEnum::Define(ref mut v) => {
                if *v != value {
                    self.const_changed = true;
                }
                *v = value;
            },
            _ => {
                if self.is_const != value {
                    self.const_changed = true;
                }
                self.is_const = value;
            }
        }
    }

    pub fn unwrap_f32(&self) -> f32 {
        match self.value {
            SettingValueEnum::F32(value, _) => value,
            _ => panic!("unwrap_f32 not F32"),
        }
    }

    pub fn unwrap_f32_mut(&mut self) -> &mut f32 {
        match self.value {
            SettingValueEnum::F32(ref mut value, _) => value,
            _ => panic!("unwrap_f32 not F32"),
        }
    }

    pub fn unwrap_u32(&self) -> u32 {
        match self.value {
            SettingValueEnum::U32(value) => value,
            _ => panic!("unwrap_u32 not U32"),
        }
    }

    pub fn format_opencl_struct(&self) -> Option<String> {
        if !self.is_const {
            match self.value {
                SettingValueEnum::F32(_, _) => {
                    Some(format!("float _{};\n", self.key))
                }
                SettingValueEnum::U32(_) => {
                    Some(format!("int _{};\n", self.key))
                }
                SettingValueEnum::Define(_) => {
                    None
                }
            }
        } else {
            None
        }
    }

    pub fn format_opencl(&self) -> String {
        let ty;
        let string;
        let is_const;
        match self.value {
            SettingValueEnum::F32(value, _) => {
                is_const = self.is_const;
                ty = "float";
                string = format!("{:.16}f", value);
            }
            SettingValueEnum::U32(value) => {
                is_const = self.is_const;
                ty = "int";
                string = format!("{}", value);
            }
            SettingValueEnum::Define(value) => {
                if value {
                    return format!("#define {} 1\n", self.key)
                } else {
                    return "".to_string();
                }
            }
        }
        if is_const {
            format!("static {} {}(__local struct MandelboxCfg const* cfg) {{ return {}; }}\n", ty, self.key, string)
        } else {
            format!("static {} {}(__local struct MandelboxCfg const* cfg) {{ return cfg->_{1}; }}\n", ty, self.key)
        }
    }
}

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
            if value.is_const {
                value.const_changed = true;
            }
            value.is_const = false;
        }
    }

    pub fn all_constants(&mut self) {
        for value in &mut self.values {
            if !value.is_const {
                value.const_changed = true;
            }
            value.is_const = true;
        }
    }

    fn get(&self, key: &str) -> Option<&SettingValue> {
        for value in &self.values {
            if value.key == key {
                return Some(value);
            }
        }
        None
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut SettingValue> {
        for value in &mut self.values {
            if value.key == key {
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
        let file = ::std::fs::File::create(file)?;
        let mut writer = ::std::io::BufWriter::new(&file);
        for value in &self.values {
            match value.value {
                SettingValueEnum::F32(v, _) => writeln!(&mut writer, "{} = {}", &value.key, v)?,
                SettingValueEnum::U32(v) => writeln!(&mut writer, "{} = {}", &value.key, v)?,
                SettingValueEnum::Define(v) => writeln!(&mut writer, "{} = {}", &value.key, v)?,
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
        for value in &self.values {
            match value.value {
                SettingValueEnum::F32(v, _) => writeln!(&mut writer, "{} = {}", &value.key, v)?,
                SettingValueEnum::U32(v) => writeln!(&mut writer, "{} = {}", &value.key, v)?,
                SettingValueEnum::Define(v) => writeln!(&mut writer, "{} = {}", &value.key, v)?,
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
            match value.value {
                SettingValueEnum::F32(_, change) => {
                    value.value = SettingValueEnum::F32(new_value.parse()?, change);
                }
                SettingValueEnum::U32(_) => {
                    value.value = SettingValueEnum::U32(new_value.parse()?);
                }
                SettingValueEnum::Define(_) => {
                    let new_value = SettingValueEnum::Define(new_value.parse()?);
                    if value.value != new_value {
                        value.const_changed = true;
                    }
                    value.value = new_value;
                }
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

    pub fn status(&self, input: &Input) -> String {
        //let mut keys = self.value_map.keys().collect::<Vec<_>>();
        //keys.sort();
        let mut builder = String::new();
        for (ind, value) in self.values.iter().enumerate() {
            let selected = if ind == input.index { "*" } else { " " };
            let constant = if value.is_const { "@" } else { " " };
            match value.value {
                SettingValueEnum::F32(v, _) => writeln!(
                    &mut builder,
                    "{}{}{} = {}",
                    selected, constant, &value.key, v
                )
                .unwrap(),
                SettingValueEnum::U32(v) => writeln!(
                    &mut builder,
                    "{}{}{} = {}",
                    selected, constant, &value.key, v
                )
                .unwrap(),
                SettingValueEnum::Define(v) => writeln!(
                    &mut builder,
                    "{}{}{} = {}",
                    selected, constant, &value.key, v
                )
                .unwrap(),
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
            result |= check(&mut value.const_changed);
        }
        result
    }

    pub fn set_src(&mut self, src: &str) {
        lazy_static! {
            //static ref RE: Regex = Regex::new(
            //   r#"(?m)^ *(?P<kind>float|int) _(?P<name>[a-zA-Z0-9_]+); *// (?P<value>[-+]?\d+(?:\.\d+)?) *(?P<change>[-+]?\d+(?:\.\d+)?)? *(?P<const>const)? *\r?$"#).unwrap();
            static ref RE: Regex = Regex::new(
               r#"(?m)^ *extern *(?P<kind>float|int) (?P<name>[a-zA-Z0-9_]+)\([^)]*\); *// *(?P<value>[-+]?\d+(?:\.\d+)?) *(?P<change>[-+]?\d+(?:\.\d+)?)? *(?P<const>const)? *\r?$"#).unwrap();
        }
        // TODO: Remove values no longer present
        let mut once = false;
        for cap in RE.captures_iter(src) {
            once = true;
            let kind = &cap["kind"];
            let name = &cap["name"];
            let setting = match kind {
                "float" => {
                    let value = cap["value"].parse().unwrap();
                    let change = cap["change"].parse().unwrap();
                    SettingValueEnum::F32(value, change)
                }
                "int" => {
                    let value = cap["value"].parse().unwrap();
                    SettingValueEnum::U32(value)
                }
                _ => {
                    panic!("Regex returned invalid kind");
                }
            };
            self.update_value(name, setting, cap.name("const").is_some());
        }
        if self.get("render_scale").is_none() {
            self.values.push(SettingValue {
                key: "render_scale".to_string(),
                value: SettingValueEnum::U32(1),
                default_value: SettingValueEnum::U32(1),
                is_const: false,
                const_changed: false,
            })
        }
        assert!(once, "Regex should get at least one setting");
        self.find_defines(src);
    }

    fn update_value(&mut self, name: &str, new_value: SettingValueEnum, is_const: bool) {
        let insert = {
            let old_value = self.get_mut(name);
            match (old_value.map(|x| x.value), new_value) {
                (
                    Some(SettingValueEnum::F32(_, ref mut old_speed)),
                    SettingValueEnum::F32(_, new_speed),
                ) => {
                    *old_speed = new_speed;
                    false
                }
                (Some(SettingValueEnum::U32(_)), SettingValueEnum::U32(_)) => false,
                _ => true,
            }
        };
        if insert {
            self.values.push(SettingValue {
                key: name.to_string(),
                value: new_value,
                default_value: new_value,
                is_const,
                const_changed: true,
            })
        }
    }

    fn find_defines(&mut self, src: &str) {
        lazy_static! {
            static ref RE: Regex =
                Regex::new(r#"(?m)^ *#ifdef +(?P<name>[a-zA-Z0-9_]+) *\r?$"#).unwrap();
        }
        for cap in RE.captures_iter(src) {
            let name = &cap["name"];
            if self.get(name).is_none() {
                let new_value = SettingValueEnum::Define(false);
                self.values.push(SettingValue {
                    key: name.to_string(),
                    value: new_value,
                    default_value: new_value,
                    is_const: false,
                    const_changed: false,
                })
            }
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::new();
        for value in &self.values {
            if !value.is_const {
                match value.value {
                    SettingValueEnum::F32(x, _) => result.write_f32::<NativeEndian>(x).unwrap(),
                    SettingValueEnum::U32(x) => result.write_u32::<NativeEndian>(x).unwrap(),
                    SettingValueEnum::Define(_) => (),
                }
            }
        }
        result
    }

    pub fn normalize(&mut self) {
        let mut look = Vector::read(self, "look_x", "look_y", "look_z");
        let mut up = Vector::read(self, "up_x", "up_y", "up_z");
        look = look.normalized();
        up = Vector::cross(Vector::cross(look, up), look).normalized();
        look.write(self, "look_x", "look_y", "look_z");
        up.write(self, "up_x", "up_y", "up_z");
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
    prev: SettingValueEnum,
    cur: SettingValueEnum,
    next: SettingValueEnum,
    next2: SettingValueEnum,
    time: f32,
    linear: bool,
) -> SettingValueEnum {
    match (prev, cur, next, next2) {
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
            let prev = self.keyframes[index_prev].find(&value.key).value;
            let cur = self.keyframes[index_cur].find(&value.key).value;
            let next = self.keyframes[index_next].find(&value.key).value;
            let next2 = self.keyframes[index_next2].find(&value.key).value;
            let result = interpolate(prev, cur, next, next2, time, self.keyframes.len() <= 2 && !wrap);
            match value.value {
                v @ SettingValueEnum::Define(_) if v != result => self.base.rebuild = true,
                _ => (),
            }
            value.value = result;
        }
        self.base.normalize();
        &mut self.base
    }
}
