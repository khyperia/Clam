use crate::texture_blit::TextureBlit;
use crate::CpuTexture;
use crate::{
    cast_slice,
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
    light_pos_1: Vec4,
    light_radius_1: f32,
    light_brightness_1_hue: f32,
    light_brightness_1_sat: f32,
    light_brightness_1_val: f32,
    ambient_brightness_hue: f32,
    ambient_brightness_sat: f32,
    ambient_brightness_val: f32,
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
    width: u32,
    height: u32,
    frame: u32,
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
    Meta::Vec3(
        "light_pos_1",
        Vector3::new(3.0, 3.5, 2.5),
        1.0,
        |s| &s.light_pos_1,
        |s| &mut s.light_pos_1,
    ),
    Meta::Float(
        "light_radius_1",
        1.0,
        -0.5,
        |s| &s.light_radius_1,
        |s| &mut s.light_radius_1,
    ),
    Meta::Float(
        "light_brightness_1_hue",
        0.0,
        0.25,
        |s| &s.light_brightness_1_hue,
        |s| &mut s.light_brightness_1_hue,
    ),
    Meta::Float(
        "light_brightness_1_sat",
        0.4,
        -1.0,
        |s| &s.light_brightness_1_sat,
        |s| &mut s.light_brightness_1_sat,
    ),
    Meta::Float(
        "light_brightness_1_val",
        4.0,
        -1.0,
        |s| &s.light_brightness_1_val,
        |s| &mut s.light_brightness_1_val,
    ),
    Meta::Float(
        "ambient_brightness_hue",
        0.65,
        0.25,
        |s| &s.ambient_brightness_hue,
        |s| &mut s.ambient_brightness_hue,
    ),
    Meta::Float(
        "ambient_brightness_sat",
        0.2,
        -1.0,
        |s| &s.ambient_brightness_sat,
        |s| &mut s.ambient_brightness_sat,
    ),
    Meta::Float(
        "ambient_brightness_val",
        0.5,
        -1.0,
        |s| &s.ambient_brightness_val,
        |s| &mut s.ambient_brightness_val,
    ),
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

struct KernelImage {
    width: u32,
    height: u32,
    scale: u32,
    img: wgpu::Texture,
    randbuf: wgpu::Texture,
    uniforms: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

fn new_tex(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsage,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth: 1,
    };
    device.create_texture(&wgpu::TextureDescriptor {
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        label: None,
    })
}

fn new_texes(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::Texture) {
    let img = new_tex(
        device,
        width,
        height,
        wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::SAMPLED,
        wgpu::TextureFormat::Rgba32Float,
    );
    let randbuf = new_tex(
        device,
        width,
        height,
        wgpu::TextureUsage::STORAGE,
        wgpu::TextureFormat::R32Uint,
    );
    (img, randbuf)
}

fn create_bind_group(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    img: &wgpu::Texture,
    randbuf: &wgpu::Texture,
    uniforms: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &img.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &randbuf.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(uniforms.slice(..)),
            },
        ],
    })
}

impl KernelImage {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        dimension: wgpu::TextureViewDimension::D2,
                        format: wgpu::TextureFormat::Rgba32Float,
                        readonly: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        dimension: wgpu::TextureViewDimension::D2,
                        format: wgpu::TextureFormat::R32Uint,
                        readonly: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let (img, randbuf) = new_texes(device, width, height);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<KernelUniforms>() as u64,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = create_bind_group(device, &bind_group_layout, &img, &randbuf, &uniforms);
        Self {
            width,
            height,
            scale: 1,
            img,
            randbuf,
            uniforms,
            bind_group_layout,
            bind_group,
        }
    }

    fn size(&self) -> (u32, u32) {
        (self.width / self.scale, self.height / self.scale)
    }

    fn resize(
        &mut self,
        device: &wgpu::Device,
        new_width: u32,
        new_height: u32,
        new_scale: u32,
    ) -> bool {
        let old_size = self.size();
        self.width = new_width;
        self.height = new_height;
        self.scale = new_scale.max(1);
        let (width, height) = self.size();
        if old_size != (width, height) {
            let (img, randbuf) = new_texes(device, width, height);
            self.img = img;
            self.randbuf = randbuf;
            self.bind_group = create_bind_group(
                device,
                &self.bind_group_layout,
                &self.img,
                &self.randbuf,
                &self.uniforms,
            );
            true
        } else {
            false
        }
    }
}

pub struct Kernel {
    kernel: wgpu::ComputePipeline,
    data: KernelImage,
    old_settings: Settings,
    frame: u32,
}

impl Kernel {
    pub fn create(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let module = device.create_shader_module(wgpu::include_spirv!("mandelbox.comp.spv"));

        let data = KernelImage::new(device, width, height);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&data.bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            compute_stage: wgpu::ProgrammableStageDescriptor {
                module: &module,
                entry_point: "main",
            },
        });

        Self {
            kernel: pipeline,
            data,
            old_settings: Settings::new(),
            frame: 0,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.data.resize(device, width, height, self.data.scale) {
            self.frame = 0;
        }
    }

    pub fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &Settings,
    ) {
        if self.data.resize(
            device,
            self.data.width,
            self.data.height,
            settings.find("render_scale").unwrap_u32() as u32,
        ) || &self.old_settings != settings
        {
            self.frame = 0;
        }
        let mut uniforms = KernelUniforms::from_settings(settings);
        let (width, height) = self.data.size();
        uniforms.width = width;
        uniforms.height = height;
        uniforms.frame = self.frame;
        let uniforms_arr = [uniforms];
        let uniforms_u8: &[u8] = cast_slice(&uniforms_arr);
        queue.write_buffer(&self.data.uniforms, 0, uniforms_u8);
        self.old_settings = settings.clone();

        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&self.kernel);
        pass.set_bind_group(0, &self.data.bind_group, &[]);
        let (width, height) = self.data.size();
        pass.dispatch(width * height, 1, 1);
        self.frame += 1;
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.data.img
    }

    pub fn download(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> CpuTexture {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let size = wgpu::Extent3d {
            width: self.data.width,
            height: self.data.height,
            depth: 1,
        };
        let texture = Self::render_to_texture(device, &mut encoder, &self.data.img, size);
        let data = Self::download_texture(device, queue, encoder, &texture, size);
        let mut result = CpuTexture {
            data,
            size: (size.width, size.height),
        };
        result.rgba_to_rgb();
        result
    }

    fn render_to_texture(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        size: wgpu::Extent3d,
    ) -> wgpu::Texture {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let dst = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsage::COPY_SRC | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        });
        let texture_blit = TextureBlit::new(device, format, &src.create_view(&Default::default()));
        texture_blit.blit(encoder, &dst.create_view(&Default::default()));
        dst
    }

    fn download_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mut encoder: wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        size: wgpu::Extent3d,
    ) -> Vec<u8> {
        let pixel_bytes = 4;
        let output_size =
            pixel_bytes * size.width as wgpu::BufferAddress * size.height as wgpu::BufferAddress;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output_size,
            usage: wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &output_buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: size.width * pixel_bytes as u32,
                    rows_per_image: size.height,
                },
            },
            size,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let output_slice = output_buffer.slice(..);
        println!("Download 1");
        let future = output_slice.map_async(wgpu::MapMode::Read);
        println!("Download 2");
        device.poll(wgpu::Maintain::Wait);
        println!("Download 3");
        futures::executor::block_on(future).unwrap();
        println!("Download 4");
        let data = output_slice.get_mapped_range().to_vec();
        println!("Download 5");
        output_buffer.unmap();
        data
    }
}
