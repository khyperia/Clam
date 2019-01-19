use ocl;
use settings::SettingValue;
use settings::Settings;

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
    fog_distance: f32,
    fog_brightness: f32,
    light_pos_1_x: f32,
    light_pos_1_y: f32,
    light_pos_1_z: f32,
    light_radius_1: f32,
    light_brightness_1_r: f32,
    light_brightness_1_g: f32,
    light_brightness_1_b: f32,
    ambient_brightness_r: f32,
    ambient_brightness_g: f32,
    ambient_brightness_b: f32,
    reflect_brightness: f32,
    surface_color_shift: f32,
    surface_color_saturation: f32,
    bailout: f32,
    de_multiplier: f32,
    max_ray_dist: f32,
    quality_first_ray: f32,
    quality_rest_ray: f32,
    white_clamp: u32,
    max_iters: u32,
    max_ray_steps: u32,
    num_ray_bounces: u32,
    speed_boost: u32,
}

unsafe impl ocl::traits::OclPrm for MandelboxCfg {}

impl MandelboxCfg {
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
            "fog_distance" => &mut self.fog_distance,
            "fog_brightness" => &mut self.fog_brightness,
            "light_pos_1_x" => &mut self.light_pos_1_x,
            "light_pos_1_y" => &mut self.light_pos_1_y,
            "light_pos_1_z" => &mut self.light_pos_1_z,
            "light_radius_1" => &mut self.light_radius_1,
            "light_brightness_1_r" => &mut self.light_brightness_1_r,
            "light_brightness_1_g" => &mut self.light_brightness_1_g,
            "light_brightness_1_b" => &mut self.light_brightness_1_b,
            "ambient_brightness_r" => &mut self.ambient_brightness_r,
            "ambient_brightness_g" => &mut self.ambient_brightness_g,
            "ambient_brightness_b" => &mut self.ambient_brightness_b,
            "reflect_brightness" => &mut self.reflect_brightness,
            "surface_color_shift" => &mut self.surface_color_shift,
            "surface_color_saturation" => &mut self.surface_color_saturation,
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
            "num_ray_bounces" => &mut self.num_ray_bounces,
            "speed_boost" => &mut self.speed_boost,
            _ => return None,
        };
        Some(val)
    }

    pub fn read(&mut self, settings: &Settings) {
        for (key, value) in settings.value_map() {
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
}
