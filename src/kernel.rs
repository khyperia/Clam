use std::fs::File;

use wgpu::util::DeviceExt;

use crate::{
    cast_slice, kernel_uniforms::KernelUniforms, settings::Settings, texture_blit::TextureBlit,
    CpuTexture,
};

struct KernelImage {
    width: u32,
    height: u32,
    scale: u32,
    img: wgpu::Texture,
    randbuf: wgpu::Texture,
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    sky: wgpu::Texture,
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

fn load_sky(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let file = File::open("Arches_E_PineTree_3k.hdr").unwrap();
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
            depth: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let image_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents,
        usage: wgpu::BufferUsage::COPY_SRC,
    });
    encoder.copy_buffer_to_texture(
        wgpu::BufferCopyView {
            buffer: &image_buf,
            layout: wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: image.width as u32 * 16,
                rows_per_image: image.height as u32,
            },
        },
        wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::Extent3d {
            width: image.width as u32,
            height: image.height as u32,
            depth: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    texture
}

fn create_bind_group(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    img: &wgpu::Texture,
    randbuf: &wgpu::Texture,
    uniforms: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    sky: &wgpu::Texture,
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::SampledTexture {
                        dimension: wgpu::TextureViewDimension::D2,
                        component_type: wgpu::TextureComponentType::Float,
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
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
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
        let module = device.create_shader_module(wgpu::include_spirv!("mandelbox.comp.spv"));

        let data = KernelImage::new(device, queue, width, height);
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
            usage: wgpu::BufferUsage::COPY_SRC,
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
        let future = output_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);
        futures::executor::block_on(future).unwrap();
        let data = output_slice.get_mapped_range().to_vec();
        output_buffer.unmap();
        data
    }
}
