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
    light_pos_2_x: f32,
    light_pos_2_y: f32,
    light_pos_2_z: f32,
    light_brightness_2_r: f32,
    light_brightness_2_g: f32,
    light_brightness_2_b: f32,
    ambient_brightness_r: f32,
    ambient_brightness_g: f32,
    ambient_brightness_b: f32,
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

pub const DEFAULT_CFG: MandelboxCfg = MandelboxCfg {
    pos_x: 0.0,
    pos_y: 0.0,
    pos_z: 5.0,
    look_x: 0.0,
    look_y: 0.0,
    look_z: -1.0,
    up_x: 0.0,
    up_y: 1.0,
    up_z: 0.0,
    fov: 1.0,
    focal_distance: 3.0,
    scale: -2.0,
    folding_limit: 1.0,
    fixed_radius_2: 1.0,
    min_radius_2: 0.125,
    dof_amount: 0.001,
    light_pos_1_x: 3.0,
    light_pos_1_y: 3.5,
    light_pos_1_z: 2.5,
    light_brightness_1_r: 5.0,
    light_brightness_1_g: 4.0,
    light_brightness_1_b: 3.0,
    light_pos_2_x: 0.0,
    light_pos_2_y: 4.0,
    light_pos_2_z: 3.0,
    light_brightness_2_r: 2.0,
    light_brightness_2_g: 1.0,
    light_brightness_2_b: 3.0,
    ambient_brightness_r: 0.3,
    ambient_brightness_g: 0.3,
    ambient_brightness_b: 0.6,
    reflect_brightness: 0.5,
    bailout: 1024.0,
    de_multiplier: 0.95,
    max_ray_dist: 16.0,
    quality_first_ray: 2.0,
    quality_rest_ray: 64.0,
    white_clamp: 1,
    max_iters: 64,
    max_ray_steps: 256,
};

impl MandelboxCfg {
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
        settings.insert("pos_x".into(), SettingValue::F32(self.pos_x, 1.0));
        settings.insert("pos_y".into(), SettingValue::F32(self.pos_y, 1.0));
        settings.insert("pos_z".into(), SettingValue::F32(self.pos_z, 1.0));
        settings.insert("look_x".into(), SettingValue::F32(self.look_x, 1.0));
        settings.insert("look_y".into(), SettingValue::F32(self.look_y, 1.0));
        settings.insert("look_z".into(), SettingValue::F32(self.look_z, 1.0));
        settings.insert("up_x".into(), SettingValue::F32(self.up_x, 1.0));
        settings.insert("up_y".into(), SettingValue::F32(self.up_y, 1.0));
        settings.insert("up_z".into(), SettingValue::F32(self.up_z, 1.0));
        settings.insert("fov".into(), SettingValue::F32(self.fov, -1.0));
        settings.insert(
            "focal_distance".into(),
            SettingValue::F32(self.focal_distance, -1.0),
        );
        settings.insert("scale".into(), SettingValue::F32(self.scale, -0.5));
        settings.insert(
            "folding_limit".into(),
            SettingValue::F32(self.folding_limit, -0.5),
        );
        settings.insert(
            "fixed_radius_2".into(),
            SettingValue::F32(self.fixed_radius_2, -0.5),
        );
        settings.insert(
            "min_radius_2".into(),
            SettingValue::F32(self.min_radius_2, -0.5),
        );
        settings.insert(
            "dof_amount".into(),
            SettingValue::F32(self.dof_amount, -0.5),
        );
        settings.insert(
            "light_pos_1_x".into(),
            SettingValue::F32(self.light_pos_1_x, 0.5),
        );
        settings.insert(
            "light_pos_1_y".into(),
            SettingValue::F32(self.light_pos_1_y, 0.5),
        );
        settings.insert(
            "light_pos_1_z".into(),
            SettingValue::F32(self.light_pos_1_z, 0.5),
        );
        settings.insert(
            "light_brightness_1_r".into(),
            SettingValue::F32(self.light_brightness_1_r, -0.5),
        );
        settings.insert(
            "light_brightness_1_g".into(),
            SettingValue::F32(self.light_brightness_1_g, -0.5),
        );
        settings.insert(
            "light_brightness_1_b".into(),
            SettingValue::F32(self.light_brightness_1_b, -0.5),
        );
        settings.insert(
            "light_pos_2_x".into(),
            SettingValue::F32(self.light_pos_2_x, 0.5),
        );
        settings.insert(
            "light_pos_2_y".into(),
            SettingValue::F32(self.light_pos_2_y, 0.5),
        );
        settings.insert(
            "light_pos_2_z".into(),
            SettingValue::F32(self.light_pos_2_z, 0.5),
        );
        settings.insert(
            "light_brightness_2_r".into(),
            SettingValue::F32(self.light_brightness_2_r, -0.5),
        );
        settings.insert(
            "light_brightness_2_g".into(),
            SettingValue::F32(self.light_brightness_2_g, -0.5),
        );
        settings.insert(
            "light_brightness_2_b".into(),
            SettingValue::F32(self.light_brightness_2_b, -0.5),
        );
        settings.insert(
            "ambient_brightness_r".into(),
            SettingValue::F32(self.ambient_brightness_r, -0.25),
        );
        settings.insert(
            "ambient_brightness_g".into(),
            SettingValue::F32(self.ambient_brightness_g, -0.25),
        );
        settings.insert(
            "ambient_brightness_b".into(),
            SettingValue::F32(self.ambient_brightness_b, -0.25),
        );
        settings.insert(
            "reflect_brightness".into(),
            SettingValue::F32(self.reflect_brightness, 0.125),
        );
        settings.insert("bailout".into(), SettingValue::F32(self.bailout, -1.0));
        settings.insert(
            "de_multiplier".into(),
            SettingValue::F32(self.de_multiplier, 0.125),
        );
        settings.insert(
            "max_ray_dist".into(),
            SettingValue::F32(self.max_ray_dist, -0.5),
        );
        settings.insert(
            "quality_first_ray".into(),
            SettingValue::F32(self.quality_first_ray, -0.5),
        );
        settings.insert(
            "quality_rest_ray".into(),
            SettingValue::F32(self.quality_rest_ray, -0.5),
        );
        settings.insert("white_clamp".into(), SettingValue::U32(self.white_clamp));
        settings.insert("max_iters".into(), SettingValue::U32(self.max_iters));
        settings.insert(
            "max_ray_steps".into(),
            SettingValue::U32(self.max_ray_steps),
        );
    }

    /*
    fn get_f32<'a, 'b>(&'a self, key: &'b str) -> Option<&'a f32> {
        let val = match key {
            "pos_x" => &self.pos_x,
            "pos_y" => &self.pos_y,
            "pos_z" => &self.pos_z,
            "look_x" => &self.look_x,
            "look_y" => &self.look_y,
            "look_z" => &self.look_z,
            "up_x" => &self.up_x,
            "up_y" => &self.up_y,
            "up_z" => &self.up_z,
            "fov" => &self.fov,
            "focal_distance" => &self.focal_distance,
            "scale" => &self.scale,
            "folding_limit" => &self.folding_limit,
            "fixed_radius_2" => &self.fixed_radius_2,
            "min_radius_2" => &self.min_radius_2,
            "dof_amount" => &self.dof_amount,
            "light_pos_x" => &self.light_pos_x,
            "light_pos_y" => &self.light_pos_y,
            "light_pos_z" => &self.light_pos_z,
            "light_brightness_r" => &self.light_brightness_r,
            "light_brightness_g" => &self.light_brightness_g,
            "light_brightness_b" => &self.light_brightness_b,
            "ambient_brightness_r" => &self.ambient_brightness_r,
            "ambient_brightness_g" => &self.ambient_brightness_g,
            "ambient_brightness_b" => &self.ambient_brightness_b,
            "reflect_brightness" => &self.reflect_brightness,
            "bailout" => &self.bailout,
            "de_multiplier" => &self.de_multiplier,
            "max_ray_dist" => &self.max_ray_dist,
            "quality_first_ray" => &self.quality_first_ray,
            "quality_rest_ray" => &self.quality_rest_ray,
            _ => return None,
        };
        Some(val)
    }

    fn get_u32<'a, 'b>(&'a self, key: &'b str) -> Option<&'a u32> {
        let val = match key {
            "white_clamp" => &self.white_clamp,
            "max_iters" => &self.max_iters,
            "max_ray_steps" => &self.max_ray_steps,
            _ => return None,
        };
        Some(val)
    }
    */

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
            "light_pos_2_x" => &mut self.light_pos_2_x,
            "light_pos_2_y" => &mut self.light_pos_2_y,
            "light_pos_2_z" => &mut self.light_pos_2_z,
            "light_brightness_2_r" => &mut self.light_brightness_2_r,
            "light_brightness_2_g" => &mut self.light_brightness_2_g,
            "light_brightness_2_b" => &mut self.light_brightness_2_b,
            "ambient_brightness_r" => &mut self.ambient_brightness_r,
            "ambient_brightness_g" => &mut self.ambient_brightness_g,
            "ambient_brightness_b" => &mut self.ambient_brightness_b,
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
            "scale" => true,
            "folding_limit" => true,
            "fixed_radius_2" => true,
            "min_radius_2" => true,
            "dof_amount" => true,
            "light_pos_1_x" => true,
            "light_pos_1_y" => true,
            "light_pos_1_z" => true,
            "light_brightness_1_r" => true,
            "light_brightness_1_g" => true,
            "light_brightness_1_b" => true,
            "light_pos_2_x" => true,
            "light_pos_2_y" => true,
            "light_pos_2_z" => true,
            "light_brightness_2_r" => true,
            "light_brightness_2_g" => true,
            "light_brightness_2_b" => true,
            "ambient_brightness_r" => true,
            "ambient_brightness_g" => true,
            "ambient_brightness_b" => true,
            "reflect_brightness" => true,
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
