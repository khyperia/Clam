use failure::Error;
use input::Input;
use mandelbox_cfg::MandelboxCfg;
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
    value_map: HashMap<String, SettingValue>,
    constants: HashSet<String>,
    rebuild: bool,
}

impl Settings {
    pub fn new() -> Self {
        let (value_map, constants) = default_settings::make_defaults();
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
        for key in self.value_map.keys() {
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

    pub fn nth(&self, index: usize) -> &'static str {
        default_settings::nth(index)
    }

    pub fn status(&self, input: &Input) -> String {
        let keys = default_settings::keys();
        //let mut keys = self.value_map.keys().collect::<Vec<_>>();
        //keys.sort();
        let mut builder = String::new();
        for (ind, key) in keys.enumerate() {
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
}

mod default_settings {
    use super::SettingValue;
    use std::collections::HashMap;
    use std::collections::HashSet;

    pub fn keys() -> impl Iterator<Item = &'static str> {
        DEFAULTS.iter().map(|&(x, _, _)| x)
    }

    pub fn make_defaults() -> (HashMap<String, SettingValue>, HashSet<String>) {
        let mut value_map = HashMap::new();
        let mut constants = HashSet::new();
        for &(key, value, is_const) in DEFAULTS.iter() {
            value_map.insert(key.to_string(), value);
            if is_const {
                constants.insert(key.to_string());
            }
        }
        (value_map, constants)
    }

    pub fn nth(index: usize) -> &'static str {
        DEFAULTS[index].0
    }

    // name, value, is_const
    const DEFAULTS: [(&str, SettingValue, bool); 41] = [
        ("pos_x", SettingValue::F32(0.0, 1.0), false),
        ("pos_y", SettingValue::F32(0.0, 1.0), false),
        ("pos_z", SettingValue::F32(5.0, 1.0), false),
        ("look_x", SettingValue::F32(0.0, 1.0), false),
        ("look_y", SettingValue::F32(0.0, 1.0), false),
        ("look_z", SettingValue::F32(-1.0, 1.0), false),
        ("up_x", SettingValue::F32(0.0, 1.0), false),
        ("up_y", SettingValue::F32(1.0, 1.0), false),
        ("up_z", SettingValue::F32(0.0, 1.0), false),
        ("fov", SettingValue::F32(1.0, -1.0), false),
        ("focal_distance", SettingValue::F32(3.0, -1.0), false),
        ("scale", SettingValue::F32(-2.0, 0.5), false),
        ("folding_limit", SettingValue::F32(1.0, -0.5), false),
        ("fixed_radius_2", SettingValue::F32(1.0, -0.5), false),
        ("min_radius_2", SettingValue::F32(0.125, -0.5), false),
        ("dof_amount", SettingValue::F32(0.001, -0.5), false),
        ("fog_distance", SettingValue::F32(1.000, -0.5), false),
        ("light_pos_1_x", SettingValue::F32(3.0, 0.5), false),
        ("light_pos_1_y", SettingValue::F32(3.5, 0.5), false),
        ("light_pos_1_z", SettingValue::F32(2.5, 0.5), false),
        ("light_radius_1", SettingValue::F32(1.0, -0.5), false),
        ("light_brightness_1_r", SettingValue::F32(5.0, -0.5), false),
        ("light_brightness_1_g", SettingValue::F32(5.0, -0.5), false),
        ("light_brightness_1_b", SettingValue::F32(4.0, -0.5), false),
        ("ambient_brightness_r", SettingValue::F32(0.8, -0.5), false),
        ("ambient_brightness_g", SettingValue::F32(0.8, -0.5), false),
        ("ambient_brightness_b", SettingValue::F32(1.0, -0.5), false),
        ("reflect_brightness", SettingValue::F32(1.0, 0.125), false),
        ("surface_color_shift", SettingValue::F32(0.0, 0.25), false),
        ("surface_color_saturation", SettingValue::F32(1.0, 0.125), false),
        ("bailout", SettingValue::F32(1024.0, -1.0), true),
        ("de_multiplier", SettingValue::F32(0.95, 0.125), true),
        ("max_ray_dist", SettingValue::F32(16.0, -0.5), true),
        ("quality_first_ray", SettingValue::F32(2.0, -0.5), true),
        ("quality_rest_ray", SettingValue::F32(64.0, -0.5), true),
        ("white_clamp", SettingValue::U32(1), true),
        ("max_iters", SettingValue::U32(64), true),
        ("max_ray_steps", SettingValue::U32(256), true),
        ("num_ray_bounces", SettingValue::U32(3), true),
        ("speed_boost", SettingValue::U32(1), true),
        ("render_scale", SettingValue::U32(1), false),
    ];
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
        let mut default = MandelboxCfg::default();
        default.read(&self.base);
        default.normalize();
        default.write(&mut self.base.value_map);
        &self.base
    }
}
