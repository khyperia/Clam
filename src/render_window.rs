use std::path::Path;

use crate::{
    fps_counter::FpsCounter, interactive::SyncInteractiveKernel, texture_blit::TextureBlit,
};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn find_font() -> Result<&'static Path, &'static str> {
    let locations: [&'static Path; 7] = [
        "C:\\Windows\\Fonts\\arial.ttf".as_ref(),
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf".as_ref(),
        "/usr/share/fonts/TTF/FiraMono-Regular.ttf".as_ref(),
        "/usr/share/fonts/TTF/DejaVuSans.ttf".as_ref(),
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf".as_ref(),
        "/Library/Fonts/Andale Mono.ttf".as_ref(),
        "/Library/Fonts/Arial Unicode.ttf".as_ref(),
    ];
    for &location in &locations {
        if location.exists() {
            return Ok(location);
        }
    }
    Err("No font found")
}

struct RenderWindow {
    surface: wgpu::Surface,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    staging_belt: wgpu::util::StagingBelt,
    glyph: wgpu_glyph::GlyphBrush<()>,
    size: winit::dpi::PhysicalSize<u32>,
    texture_blit: TextureBlit,
    fps_counter: FpsCounter,
    interactive: SyncInteractiveKernel,
}

pub fn run_headless() -> (wgpu::Device, wgpu::Queue) {
    futures::executor::block_on(run_headless_async())
}

pub async fn run_headless_async() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .unwrap();
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                    | wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                limits: wgpu::Limits::default(),
            },
            None, // Trace path
        )
        .await
        .unwrap()
}

impl RenderWindow {
    async fn new(window: &Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        println!("Got thing: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                    limits: wgpu::Limits::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let swapchain_format = surface.get_supported_formats(&adapter)[0];

        let size = window.inner_size();

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&device, &surface_config);

        let staging_belt = wgpu::util::StagingBelt::new(1024);

        let interactive = SyncInteractiveKernel::create(&device, &queue, size.width, size.height);

        let texture_blit = TextureBlit::new(
            &device,
            swapchain_format,
            &interactive.texture().create_view(&Default::default()),
        );

        let font_path = find_font().unwrap();
        let font_data = std::fs::read(font_path).unwrap();
        let font = wgpu_glyph::ab_glyph::FontArc::try_from_vec(font_data).expect("load font");
        let glyph =
            wgpu_glyph::GlyphBrushBuilder::using_font(font).build(&device, swapchain_format);

        Self {
            surface,
            surface_config,
            device,
            queue,
            staging_belt,
            glyph,
            size,
            texture_blit,
            fps_counter: FpsCounter::new(1.0),
            interactive,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.interactive
            .resize(&self.device, self.size.width, self.size.height);
    }

    fn input(&mut self, event: &WindowEvent) {
        if let WindowEvent::KeyboardInput {
            input:
                KeyboardInput {
                    state,
                    virtual_keycode: Some(key),
                    ..
                },
            ..
        } = *event
        {
            match state {
                ElementState::Pressed => self.interactive.key_down(key),
                ElementState::Released => self.interactive.key_up(key),
            }
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        if frame.suboptimal {
            println!("suboptimal");
        }
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.interactive.run(&self.device, &mut encoder);

        self.texture_blit.set_src(
            &self.device,
            &self
                .interactive
                .texture()
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
        self.texture_blit.blit(&mut encoder, &frame_view);

        self.fps_counter.tick();
        let display_text = format!(
            "{} fps\n{}",
            self.fps_counter.value(),
            self.interactive.status()
        );
        let section = wgpu_glyph::Section {
            screen_position: (10.0, 10.0),
            text: vec![wgpu_glyph::Text::new(&display_text)],
            ..Default::default()
        };
        self.glyph.queue(section);
        self.glyph
            .draw_queued(
                &self.device,
                &mut self.staging_belt,
                &mut encoder,
                &frame_view,
                self.size.width,
                self.size.height,
            )
            .unwrap();

        self.staging_belt.finish();
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        self.staging_belt.recall();

        Ok(())
    }
}

pub fn run() -> ! {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = futures::executor::block_on(RenderWindow::new(&window));

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            state.input(event);
            match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size);
                }
                _ => {}
            }
        }
        Event::RedrawRequested(_) => match state.render() {
            Ok(_) => {}
            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
            Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
            Err(wgpu::SurfaceError::Timeout) => eprintln!("Error: Timeout"),
            Err(wgpu::SurfaceError::Outdated) => eprintln!("Error: Outdated"),
        },
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            window.request_redraw();
        }
        _ => {}
    });
}