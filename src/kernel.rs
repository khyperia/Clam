use crate::{
    buffer_blit::BufferBlit, cast_slice, kernel_uniforms::KernelUniforms, settings::Settings,
    CpuTexture,
};
use std::fs::File;
use wgpu::util::DeviceExt;

struct KernelImage {
    width: u32,
    height: u32,
    scale: u32,
    img: wgpu::Buffer,
    randbuf: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    sky: wgpu::Texture,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

fn new_texes(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Buffer, wgpu::Buffer) {
    let img = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: width as u64 * height as u64 * (4 * 4),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let randbuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: width as u64 * height as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    (img, randbuf)
}

fn load_sky(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    #[cfg(target_arch = "wasm32")]
    let file = include_bytes!("../HDR_029_Sky_Cloudy_Env.hdr") as &[u8];
    #[cfg(not(target_arch = "wasm32"))]
    let file = File::open("HDR_029_Sky_Cloudy_Env.hdr").unwrap();
    let image = hdrldr::load(file).unwrap();
    let image_rgba: Vec<(f32, f32, f32, f32)> = image
        .data
        .into_iter()
        .map(|rgb| (rgb.r, rgb.g, rgb.b, 1.0))
        .collect();
    let contents = cast_slice::<(f32, f32, f32, f32), u8>(&image_rgba);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: image.width as u32,
            height: image.height as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let image_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents,
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_texture(
        wgpu::ImageCopyBuffer {
            buffer: &image_buf,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(image.width as u32 * 16),
                rows_per_image: std::num::NonZeroU32::new(image.height as u32),
            },
        },
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: image.width as u32,
            height: image.height as u32,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    texture
}

fn create_bind_group(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    img: &wgpu::Buffer,
    randbuf: &wgpu::Buffer,
    uniforms: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    sky: &wgpu::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: img,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: randbuf,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniforms,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    &sky.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
        ],
    })
}

impl KernelImage {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), // idk might be wrong
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let (img, randbuf) = new_texes(device, width, height);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<KernelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sky = load_sky(device, queue);
        let bind_group = create_bind_group(
            device,
            &bind_group_layout,
            &img,
            &randbuf,
            &uniforms,
            &sampler,
            &sky,
        );
        Self {
            width,
            height,
            scale: 1,
            img,
            randbuf,
            uniforms,
            sampler,
            sky,
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
                &self.sampler,
                &self.sky,
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
    pub fn create(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("mandelbox.wgsl"));

        let data = KernelImage::new(device, queue, width, height);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&data.bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
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

        let uniforms_staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: uniforms_u8,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(
            &uniforms_staging,
            0,
            &self.data.uniforms,
            0,
            std::mem::size_of::<KernelUniforms>() as u64,
        );

        // queue.write_buffer(&self.data.uniforms, 0, uniforms_u8);
        self.old_settings = settings.clone();

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        pass.set_pipeline(&self.kernel);
        pass.set_bind_group(0, &self.data.bind_group, &[]);
        let (width, height) = self.data.size();
        pass.dispatch_workgroups((width * height + 63) / 64, 1, 1); // TODO: Workgroups??
        self.frame += 1;
    }

    pub fn texture(&self) -> &wgpu::Buffer {
        &self.data.img
    }

    pub fn texture_size(&self) -> (u32, u32) {
        self.data.size()
    }

    pub fn download(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> CpuTexture {
        let size = self.texture_size();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let tex = Self::copy_buffer_to_texture(device, &mut encoder, &self.data.img, size);
        let buf = Self::copy_texture_to_buffer(device, &mut encoder, &tex, size);
        queue.submit(std::iter::once(encoder.finish()));
        let data = Self::download_buffer(device, queue, &buf);
        let mut result = CpuTexture {
            data,
            size: (self.data.width, self.data.height),
        };
        result.rgba_to_rgb();
        result
    }

    fn copy_buffer_to_texture(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        size: (u32, u32),
    ) -> wgpu::Texture {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let dst = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        });
        let mut texture_blit = BufferBlit::new(device, format, &src, size, false);
        texture_blit.blit(device, encoder, &dst.create_view(&Default::default()));
        dst
    }

    fn copy_texture_to_buffer(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        size: (u32, u32),
    ) -> wgpu::Buffer {
        let dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size.0 as u64 * size.1 as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &dst,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZeroU32::new(size.0 * 4),
                    rows_per_image: std::num::NonZeroU32::new(size.1),
                },
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        dst
    }

    fn download_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Vec<u8> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        wgpu::util::DownloadBuffer::read_buffer(device, queue, &buffer.slice(..), move |dl| {
            tx.send(dl.unwrap().to_vec()).unwrap()
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap()
    }
}
