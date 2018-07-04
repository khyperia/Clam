use input::Vector;
use ocl;
use settings::SettingValue;
use settings::Settings;
use std::collections::HashMap;

#[repr(C)]
#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub struct MandelboxCfg {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    look_x: f32,
    look_y: f32,
    look_z: f32,
    up_x: f32,
    up_y: f32,
    up_z: f32,
    fov: f32,
    focal_distance: f32,
    scale: f32,
    folding_limit: f32,
    fixed_radius_2: f32,
    min_radius_2: f32,
    dof_amount: f32,
    light_pos_1_x: f32,
    light_pos_1_y: f32,
    light_pos_1_z: f32,
    light_brightness_1_r: f32,
    light_brightness_1_g: f32,
    light_brightness_1_b: f32,
    light_radius_1: f32,
    ambient_brightness_r: f32,
    ambient_brightness_g: f32,
    ambient_brightness_b: f32,
    fog_distance: f32,
    fog_scatter: f32,
    reflect_brightness: f32,
    bailout: f32,
    de_multiplier: f32,
    max_ray_dist: f32,
    quality_first_ray: f32,
    quality_rest_ray: f32,
    white_clamp: u32,
    max_iters: u32,
    max_ray_steps: u32,
}

unsafe impl ocl::traits::OclPrm for MandelboxCfg {}

// name, default_val, rate_of_change
const DEFAULTS: [(&'static str, f32, f32); 37] = [
    ("pos_x", 0.0, 1.0),
    ("pos_y", 0.0, 1.0),
    ("pos_z", 5.0, 1.0),
    ("look_x", 0.0, 1.0),
    ("look_y", 0.0, 1.0),
    ("look_z", -1.0, 1.0),
    ("up_x", 0.0, 1.0),
    ("up_y", 1.0, 1.0),
    ("up_z", 0.0, 1.0),
    ("fov", 1.0, -1.0),
    ("focal_distance", 3.0, -1.0),
    ("scale", -2.0, -0.25),
    ("folding_limit", 1.0, -0.5),
    ("fixed_radius_2", 1.0, -0.5),
    ("min_radius_2", 0.125, -0.5),
    ("dof_amount", 0.001, -0.5),
    ("light_pos_1_x", 3.0, 0.5),
    ("light_pos_1_y", 3.5, 0.5),
    ("light_pos_1_z", 2.5, 0.5),
    ("light_brightness_1_r", 5.0, -0.5),
    ("light_brightness_1_g", 4.0, -0.5),
    ("light_brightness_1_b", 3.0, -0.5),
    ("light_radius_1", 1.0, -0.5),
    ("ambient_brightness_r", 0.3, -0.25),
    ("ambient_brightness_g", 0.3, -0.25),
    ("ambient_brightness_b", 0.6, -0.25),
    ("fog_distance", 50.0, -0.5),
    ("fog_scatter", 1.0, 0.25),
    ("reflect_brightness", 0.5, 0.125),
    ("bailout", 1024.0, -1.0),
    ("de_multiplier", 0.95, 0.125),
    ("max_ray_dist", 16.0, -0.5),
    ("quality_first_ray", 2.0, -0.5),
    ("quality_rest_ray", 64.0, -0.5),
    ("white_clamp", 1.0, 0.0),
    ("max_iters", 64.0, 0.0),
    ("max_ray_steps", 256.0, 0.0),
];

impl MandelboxCfg {
    pub fn get_default() -> MandelboxCfg {
        let mut result = Self::default();
        for &(name, value, _) in DEFAULTS.iter() {
            if let Some(val) = result.get_f32_mut(name) {
                *val = value;
            }
            if let Some(val) = result.get_u32_mut(name) {
                *val = value as u32;
            }
        }
        result
    }

    pub fn num_keys() -> usize {
        DEFAULTS.len()
    }

    pub fn keys() -> impl Iterator<Item = &'static str> {
        DEFAULTS.iter().map(|&(x, _, _)| x)
    }

    fn get_f32_mut<'a, 'b>(&'a mut self, key: &'b str) -> Option<&'a mut f32> {
        let val = match key {
            "pos_x" => &mut self.pos_x,
            "pos_y" => &mut self.pos_y,
            "pos_z" => &mut self.pos_z,
            "look_x" => &mut self.look_x,
            "look_y" => &mut self.look_y,
            "look_z" => &mut self.look_z,
            "up_x" => &mut self.up_x,
            "up_y" => &mut self.up_y,
            "up_z" => &mut self.up_z,
            "fov" => &mut self.fov,
            "focal_distance" => &mut self.focal_distance,
            "scale" => &mut self.scale,
            "folding_limit" => &mut self.folding_limit,
            "fixed_radius_2" => &mut self.fixed_radius_2,
            "min_radius_2" => &mut self.min_radius_2,
            "dof_amount" => &mut self.dof_amount,
            "light_pos_1_x" => &mut self.light_pos_1_x,
            "light_pos_1_y" => &mut self.light_pos_1_y,
            "light_pos_1_z" => &mut self.light_pos_1_z,
            "light_brightness_1_r" => &mut self.light_brightness_1_r,
            "light_brightness_1_g" => &mut self.light_brightness_1_g,
            "light_brightness_1_b" => &mut self.light_brightness_1_b,
            "light_radius_1" => &mut self.light_radius_1,
            "ambient_brightness_r" => &mut self.ambient_brightness_r,
            "ambient_brightness_g" => &mut self.ambient_brightness_g,
            "ambient_brightness_b" => &mut self.ambient_brightness_b,
            "fog_distance" => &mut self.fog_distance,
            "fog_scatter" => &mut self.fog_scatter,
            "reflect_brightness" => &mut self.reflect_brightness,
            "bailout" => &mut self.bailout,
            "de_multiplier" => &mut self.de_multiplier,
            "max_ray_dist" => &mut self.max_ray_dist,
            "quality_first_ray" => &mut self.quality_first_ray,
            "quality_rest_ray" => &mut self.quality_rest_ray,
            _ => return None,
        };
        Some(val)
    }

    fn get_u32_mut<'a, 'b>(&'a mut self, key: &'b str) -> Option<&'a mut u32> {
        let val = match key {
            "white_clamp" => &mut self.white_clamp,
            "max_iters" => &mut self.max_iters,
            "max_ray_steps" => &mut self.max_ray_steps,
            _ => return None,
        };
        Some(val)
    }

    pub fn read(&mut self, settings: &Settings) {
        for (key, value) in settings.value_map() {
            if settings.is_const(key) {
                continue;
            }
            match *value {
                SettingValue::F32(new, _) => {
                    if let Some(old) = self.get_f32_mut(key) {
                        *old = new;
                    }
                }
                SettingValue::U32(new) => {
                    if let Some(old) = self.get_u32_mut(key) {
                        *old = new;
                    }
                }
            }
        }
    }

    pub fn normalize(&mut self) {
        let mut look = Vector::new(self.look_x, self.look_y, self.look_z);
        let mut up = Vector::new(self.up_x, self.up_y, self.up_z);
        look = look.normalized();
        up = Vector::cross(Vector::cross(look, up), look).normalized();
        self.look_x = look.x;
        self.look_y = look.y;
        self.look_z = look.z;
        self.up_x = up.x;
        self.up_y = up.y;
        self.up_z = up.z;
    }

    pub fn write(&mut self, settings: &mut HashMap<String, SettingValue>) {
        for &(name, _, rate_of_change) in DEFAULTS.iter() {
            if let Some(&mut val) = self.get_f32_mut(name) {
                settings.insert(name.into(), SettingValue::F32(val, rate_of_change));
            }
            if let Some(&mut val) = self.get_u32_mut(name) {
                settings.insert(name.into(), SettingValue::U32(val));
            }
        }
    }

    pub fn is_const(key: &str) -> bool {
        match key {
            "pos_x" => false,
            "pos_y" => false,
            "pos_z" => false,
            "look_x" => false,
            "look_y" => false,
            "look_z" => false,
            "up_x" => false,
            "up_y" => false,
            "up_z" => false,
            "fov" => false,
            "focal_distance" => false,
            "scale" => false,
            "folding_limit" => false,
            "fixed_radius_2" => false,
            "min_radius_2" => false,
            "dof_amount" => false,
            "light_pos_1_x" => false,
            "light_pos_1_y" => false,
            "light_pos_1_z" => false,
            "light_brightness_1_r" => false,
            "light_brightness_1_g" => false,
            "light_brightness_1_b" => false,
            "light_radius_1" => false,
            "ambient_brightness_r" => false,
            "ambient_brightness_g" => false,
            "ambient_brightness_b" => false,
            "fog_distance" => false,
            "fog_scatter" => false,
            "reflect_brightness" => false,
            "bailout" => true,
            "de_multiplier" => true,
            "max_ray_dist" => true,
            "quality_first_ray" => true,
            "quality_rest_ray" => true,
            "white_clamp" => true,
            "max_iters" => true,
            "max_ray_steps" => true,
            _ => false,
        }
    }
}
