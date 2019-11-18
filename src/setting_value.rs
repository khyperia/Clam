use cgmath::Vector3;

#[derive(Debug, PartialEq, Clone)]
pub struct SettingValue {
    key: String,
    value: SettingValueEnum,
    default_value: SettingValueEnum,
    is_const: bool,
    needs_rebuild: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub enum SettingValueEnum {
    Int(u64),
    Float(f64, f64),
    Vec3(Vector3<f64>, f64),
    Define(bool),
}

impl SettingValue {
    pub fn new(key: String, value: SettingValueEnum, is_const: bool) -> Self {
        Self {
            key,
            value: value.clone(),
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
        } else if self.is_const && self.value != value {
            self.needs_rebuild = true;
        }
        self.value = value
    }

    pub fn set_default_value(&mut self, value: SettingValueEnum) {
        self.default_value = value
    }

    pub fn change_one(&mut self, increase: bool) {
        match self.value {
            SettingValueEnum::Float(_, _) => (),
            SettingValueEnum::Vec3(_, _) => (),
            SettingValueEnum::Int(ref mut value) => {
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

    pub fn change(&mut self, increase: bool, mut dt: f64) {
        dt *= if increase { 1.0 } else { -1.0 };
        match self.value {
            SettingValueEnum::Float(ref mut value, change) => {
                if change < 0.0 {
                    *value *= (-change + 1.0).powf(dt);
                } else {
                    *value += dt * change;
                }
            }
            SettingValueEnum::Vec3(_, _) => {
                // TODO
            }
            SettingValueEnum::Define(_) | SettingValueEnum::Int(_) => (),
        }
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
                SettingValueEnum::Int(ref mut v) => *v = 0,
                SettingValueEnum::Float(ref mut v, _) => *v = 0.0,
                SettingValueEnum::Vec3(ref mut v, _) => *v = Vector3::new(0.0, 0.0, 0.0),
                SettingValueEnum::Define(ref mut v) => *v = false,
            }
        } else {
            self.value = self.default_value.clone();
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

    pub fn unwrap_u32(&self) -> u64 {
        match self.value {
            SettingValueEnum::Int(value) => value,
            _ => panic!("unwrap_u32 not U32"),
        }
    }

    pub fn unwrap_f32(&self) -> f64 {
        match self.value {
            SettingValueEnum::Float(value, _) => value,
            _ => panic!("unwrap_f32 not F32"),
        }
    }

    #[cfg(feature = "vr")]
    pub fn unwrap_f32_mut(&mut self) -> &mut f64 {
        match self.value {
            SettingValueEnum::Float(ref mut value, _) => value,
            _ => panic!("unwrap_f32 not F32"),
        }
    }

    pub fn unwrap_vec3(&self) -> Vector3<f64> {
        match self.value {
            SettingValueEnum::Vec3(value, _) => value,
            _ => panic!("unwrap_vec3 not vec3"),
        }
    }

    pub fn unwrap_vec3_mut(&mut self) -> &mut Vector3<f64> {
        match self.value {
            SettingValueEnum::Vec3(ref mut value, _) => value,
            _ => panic!("unwrap_vec3 not vec3"),
        }
    }

    #[cfg(feature = "vr")]
    pub fn unwrap_define_mut(&mut self) -> &mut bool {
        match self.value {
            SettingValueEnum::Define(ref mut value) => value,
            _ => panic!("unwrap_f32 not F32"),
        }
    }

    pub fn format_glsl(&self, src: &mut String) -> Option<String> {
        match self.value {
            SettingValueEnum::Int(x) => {
                if self.is_const() {
                    let replacement = format!("#define {} {}", self.key(), x);
                    *src = src.replace(&format!("uniform uint {};", self.key()), &replacement);
                }
                None
            }
            SettingValueEnum::Float(x, _) => {
                if self.is_const() {
                    let replacement = format!("#define {} {:.16}", self.key(), x);
                    *src = src.replace(&format!("uniform float {};", self.key()), &replacement);
                }
                None
            }
            SettingValueEnum::Vec3(x, _) => {
                if self.is_const() {
                    let replacement = format!(
                        "#define {} vec3({:.16}, {:.16}, {:.16})",
                        self.key(),
                        x.x,
                        x.y,
                        x.z
                    );
                    *src = src.replace(&format!("uniform vec3 {};", self.key()), &replacement);
                }
                None
            }
            SettingValueEnum::Define(val) => {
                if val {
                    Some(format!("#define {} 1", self.key()))
                } else {
                    None
                }
            }
        }
    }
}
