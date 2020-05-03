use crate::{
    kernel_compilation::ComputeShader, setting_value::SettingValueEnum, settings::Settings,
};
use std::fmt::Write;

pub struct SettingsInput {
    index: usize,
    component: usize,
}

impl SettingsInput {
    pub fn new() -> Self {
        Self {
            index: 0,
            component: 0,
        }
    }

    fn valid(index: usize, settings: &Settings, kernel: &Option<ComputeShader>) -> bool {
        if let Some(kernel) = kernel {
            let value = &settings.values[index];
            if let SettingValueEnum::Define(_) = value.value() {
                return true;
            }
            let current_name = value.key();
            if current_name == "render_scale" {
                return true;
            }
            for uniform in kernel.uniforms() {
                if uniform.name == current_name {
                    return true;
                }
            }
            false
        } else {
            true
        }
    }

    pub fn status(&self, settings: &Settings, kernel: &Option<ComputeShader>) -> String {
        let mut builder = String::new();
        for (ind, value) in settings.values.iter().enumerate() {
            let valid = Self::valid(ind, settings, kernel);
            if !valid {
                continue;
            }
            let selected = if ind == self.index { "*" } else { " " };
            let key = value.key();
            match value.value() {
                SettingValueEnum::Int(v) => {
                    writeln!(&mut builder, "{} {} = {}", selected, key, v).unwrap()
                }
                SettingValueEnum::Float(v, _) => {
                    writeln!(&mut builder, "{} {} = {}", selected, key, v).unwrap()
                }
                SettingValueEnum::Vec3(v, _) => {
                    let selected = if ind == self.index {
                        match self.component {
                            0 => "x",
                            1 => "y",
                            2 => "z",
                            _ => panic!("Invalid component index"),
                        }
                    } else {
                        " "
                    };
                    writeln!(
                        &mut builder,
                        "{} {} = {} {} {}",
                        selected, key, v.x, v.y, v.z
                    )
                    .unwrap()
                }
                SettingValueEnum::Define(v) => {
                    writeln!(&mut builder, "{} {} = {}", selected, key, v).unwrap()
                }
            }
        }
        builder
    }

    fn num_components_at_current(&self, settings: &Settings) -> usize {
        match settings.values[self.index].value() {
            SettingValueEnum::Vec3(_, _) => 3,
            _ => 1,
        }
    }

    fn next(&mut self, settings: &Settings, kernel: &Option<ComputeShader>, delta: isize) {
        let start = (self.index, self.component);
        assert!(delta == -1 || delta == 1);
        loop {
            let num_components_at_current = self.num_components_at_current(settings);
            if delta < 0 && self.component > 0 {
                self.component -= 1;
            } else if delta > 0 && self.component + 1 < num_components_at_current {
                self.component += 1;
            } else {
                self.index = (self.index as isize + delta)
                    .rem_euclid(settings.values.len() as isize)
                    as usize;
                self.component = if delta < 0 {
                    self.num_components_at_current(settings) - 1
                } else {
                    0
                };
            }
            // don't infinite loop if all are invalid
            if Self::valid(self.index, settings, kernel) || start == (self.index, self.component) {
                break;
            }
        }
    }

    pub fn up_one(&mut self, settings: &Settings, kernel: &Option<ComputeShader>) {
        self.next(settings, kernel, -1)
    }

    pub fn down_one(&mut self, settings: &Settings, kernel: &Option<ComputeShader>) {
        self.next(settings, kernel, 1)
    }

    pub fn left_one(&mut self, settings: &mut Settings) {
        settings.values[self.index].change_one(false)
    }

    pub fn right_one(&mut self, settings: &mut Settings) {
        settings.values[self.index].change_one(true)
    }

    pub fn toggle(&mut self, settings: &mut Settings) {
        settings.values[self.index].toggle()
    }

    pub fn left_hold(&mut self, settings: &mut Settings, dt: f64) {
        settings.values[self.index].change(self.component, false, dt);
    }

    pub fn right_hold(&mut self, settings: &mut Settings, dt: f64) {
        settings.values[self.index].change(self.component, true, dt);
    }
}
