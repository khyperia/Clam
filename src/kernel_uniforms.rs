use crate::{
    setting_value::{SettingValue, SettingValueEnum},
    settings::Settings,
};
use cgmath::Vector3;
use glam::Vec4;

#[repr(C)]
#[derive(Default)]
pub struct KernelUniforms {
    pos: Vec4,
    look: Vec4,
    up: Vec4,
    fov: f32,
    focal_distance: f32,
    scale: f32,
    folding_limit: f32,
    fixed_radius_2: f32,
    min_radius_2: f32,
    dof_amount: f32,
    bloom_amount: f32,
    bloom_size: f32,
    fog_distance: f32,
    fog_brightness: f32,
    exposure: f32,
    surface_color_variance: f32,
    surface_color_shift: f32,
    surface_color_saturation: f32,
    surface_color_value: f32,
    surface_color_gloss: f32,
    plane: Vec4,
    rotation: f32,
    bailout: f32,
    bailout_normal: f32,
    de_multiplier: f32,
    max_ray_dist: f32,
    quality_first_ray: f32,
    quality_rest_ray: f32,
    gamma: f32,
    fov_left: f32,
    fov_right: f32,
    fov_top: f32,
    fov_bottom: f32,
    max_iters: u32,
    max_ray_steps: u32,
    num_ray_bounces: u32,
    gamma_test: u32,
    pub width: u32,
    pub height: u32,
    pub frame: u32,
}

enum Meta {
    Int(
        &'static str,
        u64,
        fn(&KernelUniforms) -> &u32,
        fn(&mut KernelUniforms) -> &mut u32,
    ),
    Float(
        &'static str,
        f64,
        f64,
        fn(&KernelUniforms) -> &f32,
        fn(&mut KernelUniforms) -> &mut f32,
    ),
    Vec3(
        &'static str,
        Vector3<f64>,
        f64,
        fn(&KernelUniforms) -> &Vec4,
        fn(&mut KernelUniforms) -> &mut Vec4,
    ),
}

const UNIFORM_METADATA: &[Meta] = &[
    Meta::Vec3(
        "pos",
        Vector3::new(0.0, 0.0, 5.0),
        1.0,
        |s| &s.pos,
        |s| &mut s.pos,
    ),
    Meta::Vec3(
        "look",
        Vector3::new(0.0, 0.0, -1.0),
        1.0,
        |s| &s.look,
        |s| &mut s.look,
    ),
    Meta::Vec3(
        "up",
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        |s| &s.up,
        |s| &mut s.up,
    ),
    Meta::Float("fov", 1.0, -1.0, |s| &s.fov, |s| &mut s.fov),
    Meta::Float(
        "focal_distance",
        3.0,
        -1.0,
        |s| &s.focal_distance,
        |s| &mut s.focal_distance,
    ),
    Meta::Float("scale", -2.0, 0.5, |s| &s.scale, |s| &mut s.scale),
    Meta::Float(
        "folding_limit",
        1.0,
        -0.5,
        |s| &s.folding_limit,
        |s| &mut s.folding_limit,
    ),
    Meta::Float(
        "fixed_radius_2",
        1.0,
        -0.5,
        |s| &s.fixed_radius_2,
        |s| &mut s.fixed_radius_2,
    ),
    Meta::Float(
        "min_radius_2",
        0.125,
        -0.5,
        |s| &s.min_radius_2,
        |s| &mut s.min_radius_2,
    ),
    Meta::Float(
        "dof_amount",
        0.01,
        -1.0,
        |s| &s.dof_amount,
        |s| &mut s.dof_amount,
    ),
    Meta::Float(
        "bloom_amount",
        0.1,
        -0.25,
        |s| &s.bloom_amount,
        |s| &mut s.bloom_amount,
    ),
    Meta::Float(
        "bloom_size",
        0.01,
        -0.25,
        |s| &s.bloom_size,
        |s| &mut s.bloom_size,
    ),
    Meta::Float(
        "fog_distance",
        10.0,
        -1.0,
        |s| &s.fog_distance,
        |s| &mut s.fog_distance,
    ),
    Meta::Float(
        "fog_brightness",
        1.0,
        -0.5,
        |s| &s.fog_brightness,
        |s| &mut s.fog_brightness,
    ),
    Meta::Float("exposure", 1.0, -0.5, |s| &s.exposure, |s| &mut s.exposure),
    Meta::Float(
        "surface_color_variance",
        0.0625,
        -0.25,
        |s| &s.surface_color_variance,
        |s| &mut s.surface_color_variance,
    ),
    Meta::Float(
        "surface_color_shift",
        0.0,
        0.125,
        |s| &s.surface_color_shift,
        |s| &mut s.surface_color_shift,
    ),
    Meta::Float(
        "surface_color_saturation",
        0.75,
        0.125,
        |s| &s.surface_color_saturation,
        |s| &mut s.surface_color_saturation,
    ),
    Meta::Float(
        "surface_color_value",
        1.0,
        0.125,
        |s| &s.surface_color_value,
        |s| &mut s.surface_color_value,
    ),
    Meta::Float(
        "surface_color_gloss",
        0.0,
        0.25,
        |s| &s.surface_color_gloss,
        |s| &mut s.surface_color_gloss,
    ),
    Meta::Vec3(
        "plane",
        Vector3::new(3.0, 3.5, 2.5),
        1.0,
        |s| &s.plane,
        |s| &mut s.plane,
    ),
    Meta::Float("rotation", 0.0, 0.125, |s| &s.rotation, |s| &mut s.rotation),
    Meta::Float("bailout", 64.0, -0.25, |s| &s.bailout, |s| &mut s.bailout),
    Meta::Float(
        "bailout_normal",
        1024.0,
        -1.0,
        |s| &s.bailout_normal,
        |s| &mut s.bailout_normal,
    ),
    Meta::Float(
        "de_multiplier",
        0.9375,
        0.125,
        |s| &s.de_multiplier,
        |s| &mut s.de_multiplier,
    ),
    Meta::Float(
        "max_ray_dist",
        16.0,
        -0.5,
        |s| &s.max_ray_dist,
        |s| &mut s.max_ray_dist,
    ),
    Meta::Float(
        "quality_first_ray",
        2.0,
        -0.5,
        |s| &s.quality_first_ray,
        |s| &mut s.quality_first_ray,
    ),
    Meta::Float(
        "quality_rest_ray",
        64.0,
        -0.5,
        |s| &s.quality_rest_ray,
        |s| &mut s.quality_rest_ray,
    ),
    Meta::Float("gamma", 0.0, 0.25, |s| &s.gamma, |s| &mut s.gamma),
    Meta::Float("fov_left", -1.0, 1.0, |s| &s.fov_left, |s| &mut s.fov_left),
    Meta::Float(
        "fov_right",
        1.0,
        1.0,
        |s| &s.fov_right,
        |s| &mut s.fov_right,
    ),
    Meta::Float("fov_top", 1.0, 1.0, |s| &s.fov_top, |s| &mut s.fov_top),
    Meta::Float(
        "fov_bottom",
        -1.0,
        1.0,
        |s| &s.fov_bottom,
        |s| &mut s.fov_bottom,
    ),
    Meta::Int("max_iters", 20, |s| &s.max_iters, |s| &mut s.max_iters),
    Meta::Int(
        "max_ray_steps",
        256,
        |s| &s.max_ray_steps,
        |s| &mut s.max_ray_steps,
    ),
    Meta::Int(
        "num_ray_bounces",
        4,
        |s| &s.num_ray_bounces,
        |s| &mut s.num_ray_bounces,
    ),
    Meta::Int("gamma_test", 0, |s| &s.gamma_test, |s| &mut s.gamma_test),
];

impl KernelUniforms {
    pub fn from_settings(settings: &Settings) -> Self {
        let mut result = KernelUniforms::default();
        for m in UNIFORM_METADATA {
            match m {
                Meta::Int(name, _, _, get_mut) => {
                    *get_mut(&mut result) = settings.get(name).unwrap().unwrap_u32() as u32;
                }
                Meta::Float(name, _, _, _, get_mut) => {
                    *get_mut(&mut result) = settings.get(name).unwrap().unwrap_float() as f32;
                }
                Meta::Vec3(name, _, _, _, get_mut) => {
                    let v = settings.get(name).unwrap().unwrap_vec3();
                    *get_mut(&mut result) = Vec4::new(v.x as f32, v.y as f32, v.z as f32, 0.0);
                }
            }
        }
        result
    }

    pub fn fill_defaults(settings: &mut Settings) {
        for m in UNIFORM_METADATA {
            match *m {
                Meta::Int(name, default, _, _) => {
                    let setting = SettingValueEnum::Int(default);
                    settings
                        .values
                        .push(SettingValue::new(name.to_string(), setting));
                }
                Meta::Float(name, default, change, _, _) => {
                    let setting = SettingValueEnum::Float(default, change);
                    settings
                        .values
                        .push(SettingValue::new(name.to_string(), setting));
                }
                Meta::Vec3(name, default, change, _, _) => {
                    let setting = SettingValueEnum::Vec3(default, change);
                    settings
                        .values
                        .push(SettingValue::new(name.to_string(), setting));
                }
            }
        }
    }
}
