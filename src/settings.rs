use std::collections::HashMap;

pub enum SettingValue {
    U32(u32),
    F32(f32),
}

pub type Settings = HashMap<String, SettingValue>;
