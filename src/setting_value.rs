use crate::kernel_compilation::Uniform;
use cgmath::Vector3;
use failure::Error;

#[derive(Debug, PartialEq, Clone)]
pub struct SettingValue {
    key: String,
    value: SettingValueEnum,
    default_value: SettingValueEnum,
    is_const: bool,
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
        }
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn value(&self) -> &SettingValueEnum {
        &self.value
    }

    pub fn set_value(&mut self, value: SettingValueEnum) {
        self.value = value
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
    }

    pub fn is_const(&self) -> bool {
        self.is_const
    }

    pub fn set_const(&mut self, value: bool) {
        if let SettingValueEnum::Define(_) = self.value {
            return;
        }
        self.is_const = value;
    }

    pub fn unwrap_u32(&self) -> u64 {
        match self.value {
            SettingValueEnum::Int(value) => value,
            _ => panic!("unwrap_u32 not U32"),
        }
    }

    pub fn unwrap_float(&self) -> f64 {
        match self.value {
            SettingValueEnum::Float(value, _) => value,
            _ => panic!("unwrap_float not float"),
        }
    }

    //#[cfg(feature = "vr")]
    pub fn unwrap_float_mut(&mut self) -> &mut f64 {
        match self.value {
            SettingValueEnum::Float(ref mut value, _) => value,
            _ => panic!("unwrap_float not float"),
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
            _ => panic!("unwrap_define not define"),
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

impl SettingValueEnum {
    pub fn is_define(&self) -> bool {
        match self {
            SettingValueEnum::Define(_) => true,
            _ => false,
        }
    }

    pub fn kinds_match(&self, other: &SettingValueEnum) -> bool {
        match (self, other) {
            (SettingValueEnum::Int(_), SettingValueEnum::Int(_)) => true,
            (SettingValueEnum::Float(_, _), SettingValueEnum::Float(_, _)) => true,
            (SettingValueEnum::Vec3(_, _), SettingValueEnum::Vec3(_, _)) => true,
            (SettingValueEnum::Define(_), SettingValueEnum::Define(_)) => true,
            _ => false,
        }
    }

    pub fn set_uniform(&self, uniform: &Uniform) -> Result<(), Error> {
        match *self {
            SettingValueEnum::Int(x) => uniform.set_arg_u32(x as u32),
            SettingValueEnum::Float(x, _) => uniform.set_arg_f32(x as f32),
            SettingValueEnum::Vec3(x, _) => {
                uniform.set_arg_f32_3(x.x as f32, x.y as f32, x.z as f32)
            }
            _ => panic!("Unknown variable type in set_uniform"),
        }
    }
}
