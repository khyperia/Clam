#[derive(Debug, PartialEq, Clone)]
pub struct SettingValue {
    key: String,
    value: SettingValueEnum,
    default_value: SettingValueEnum,
    is_const: bool,
    needs_rebuild: bool,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SettingValueEnum {
    U32(u32),
    F32(f32, f32),
    Define(bool),
}

impl SettingValue {
    pub fn new(key: String, value: SettingValueEnum, is_const: bool) -> Self {
        Self {
            key,
            value,
            default_value: value,
            is_const,
            needs_rebuild: false,
        }
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn value(&self) -> &SettingValueEnum {
        &self.value
    }

    pub fn set_value(&mut self, value: SettingValueEnum) {
        if let SettingValueEnum::Define(new_val) = value {
            if let SettingValueEnum::Define(old_val) = self.value {
                self.needs_rebuild = old_val != new_val;
            } else {
                self.needs_rebuild = true;
            }
        }
        self.value = value
    }

    pub fn set_default_value(&mut self, value: SettingValueEnum) {
        self.default_value = value
    }

    pub fn change_one(&mut self, increase: bool) {
        match self.value {
            SettingValueEnum::F32(_, _) => (),
            SettingValueEnum::U32(ref mut value) => {
                if increase {
                    *value += 1;
                } else if *value != 0 {
                    *value -= 1;
                }
            }
            SettingValueEnum::Define(ref mut value) => {
                *value = !*value;
                self.needs_rebuild = true;
            }
        }
    }

    pub fn change(&mut self, increase: bool, mut dt: f32) {
        dt *= if increase { 1.0 } else { -1.0 };
        if let SettingValueEnum::F32(ref mut value, change) = self.value {
            if change < 0.0 {
                *value *= (-change + 1.0).powf(dt);
            } else {
                *value += dt * change;
            }
        };
    }

    pub fn toggle(&mut self) {
        fn def(val: &SettingValueEnum) -> Option<bool> {
            match *val {
                SettingValueEnum::Define(v) => Some(v),
                _ => None,
            }
        }
        let old_define = def(&self.value);
        if self.value == self.default_value {
            match self.value {
                SettingValueEnum::U32(ref mut v) => *v = 0,
                SettingValueEnum::F32(ref mut v, _) => *v = 0.0,
                SettingValueEnum::Define(ref mut v) => *v = false,
            }
        } else {
            self.value = self.default_value;
        }
        let new_define = def(&self.value);
        if old_define != new_define {
            self.needs_rebuild = true;
        }
    }

    pub fn is_const(&self) -> bool {
        self.is_const
    }

    pub fn set_const(&mut self, value: bool) {
        if let SettingValueEnum::Define(_) = self.value {
            return;
        }
        if self.is_const != value {
            self.needs_rebuild = true;
        }
        self.is_const = value;
    }

    pub fn check_reset_needs_rebuild(&mut self) -> bool {
        let result = self.needs_rebuild;
        self.needs_rebuild = false;
        result
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

    #[cfg(feature = "vr")]
    pub fn unwrap_define_mut(&mut self) -> &mut bool {
        match self.value {
            SettingValueEnum::Define(ref mut value) => value,
            _ => panic!("unwrap_f32 not F32"),
        }
    }

    pub fn format_opencl_struct(&self) -> Option<String> {
        if !self.is_const {
            match self.value {
                SettingValueEnum::F32(_, _) => Some(format!("float _{};\n", self.key)),
                SettingValueEnum::U32(_) => Some(format!("int _{};\n", self.key)),
                SettingValueEnum::Define(_) => None,
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
                    return format!("#define {} 1\n", self.key);
                } else {
                    return "".to_string();
                }
            }
        }
        if is_const {
            format!(
                "static {} {}(__local struct MandelboxCfg const* cfg) {{ return {}; }}\n",
                ty, self.key, string
            )
        } else {
            format!(
                "static {} {}(__local struct MandelboxCfg const* cfg) {{ return cfg->_{1}; }}\n",
                ty, self.key
            )
        }
    }
}
