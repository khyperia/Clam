use wgpu::util::DeviceExt;

use crate::cast_slice;

#[repr(C)]
#[derive(Default, Clone, Copy, PartialEq, Eq)]
struct Uniforms {
    width: u32,
    height: u32,
    output_srgb: u32,
    dummy: u32,
}

pub struct BufferBlit {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    uniforms_buf: wgpu::Buffer,
    uniforms_data: Uniforms,
    uniforms: Uniforms,
}

impl BufferBlit {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        src: &wgpu::Buffer,
        size: (u32, u32),
        output_srgb: bool,
    ) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = Self::create_bind_group(device, &bind_group_layout, src, &uniforms_buf);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let module = device.create_shader_module(wgpu::include_wgsl!("buffer_blit.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vert",
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "frag",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            uniforms_buf,
            uniforms_data: Default::default(),
            uniforms: Uniforms {
                width: size.0,
                height: size.1,
                output_srgb: u32::from(output_srgb),
                dummy: 0,
            },
        }
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        src: &wgpu::Buffer,
        sizebuf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: src,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: sizebuf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }

    pub fn set_src(
        &mut self,
        device: &wgpu::Device,
        src: &wgpu::Buffer,
        size: (u32, u32),
        output_srgb: Option<bool>,
    ) {
        self.uniforms.width = size.0;
        self.uniforms.height = size.1;
        if let Some(output_srgb) = output_srgb {
            self.uniforms.output_srgb = u32::from(output_srgb);
        }
        self.bind_group =
            Self::create_bind_group(device, &self.bind_group_layout, src, &self.uniforms_buf);
    }

    fn set_uniforms(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if self.uniforms == self.uniforms_data {
            return;
        }
        self.uniforms_data = self.uniforms;

        let slice = [self.uniforms];
        let slice_u8: &[u8] = cast_slice(&slice);
        let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: slice_u8,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(
            &staging,
            0,
            &self.uniforms_buf,
            0,
            std::mem::size_of::<Uniforms>() as u64,
        );
    }

    pub fn blit(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::TextureView,
    ) {
        self.set_uniforms(device, encoder);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..4, 0..1);
    }
}
