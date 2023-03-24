use log::{error, info, warn};

#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;

use crate::{buffer_blit::BufferBlit, fps_counter::FpsCounter, interactive::SyncInteractiveKernel};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[cfg(not(target_arch = "wasm32"))]
fn find_font() -> Result<&'static std::path::Path, &'static str> {
    let locations: [&'static std::path::Path; 7] = [
        "C:\\Windows\\Fonts\\arial.ttf".as_ref(),
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf".as_ref(),
        "/usr/share/fonts/TTF/FiraMono-Regular.ttf".as_ref(),
        "/usr/share/fonts/TTF/DejaVuSans.ttf".as_ref(),
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf".as_ref(),
        "/Library/Fonts/Andale Mono.ttf".as_ref(),
        "/Library/Fonts/Arial Unicode.ttf".as_ref(),
    ];
    for location in locations {
        if location.exists() {
            return Ok(location);
        }
    }
    Err("No font found")
}

pub struct RenderWindow {
    event_loop: Option<EventLoop<()>>,
    window: Window,
    surface: wgpu::Surface,
    swapchain_format: wgpu::TextureFormat,
    device: wgpu::Device,
    queue: wgpu::Queue,
    staging_belt: wgpu::util::StagingBelt,
    glyph: wgpu_text::TextBrush,
    size: winit::dpi::PhysicalSize<u32>,
    buffer_blit: BufferBlit,
    fps_counter: FpsCounter,
    interactive: SyncInteractiveKernel,
}

pub async fn run_headless() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        dx12_shader_compiler: Default::default(),
    });
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
    pub async fn new() -> Result<Self, ()> {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new().build(&event_loop).unwrap();

        #[cfg(target_arch = "wasm32")]
        {
            if matches!(window.canvas().get_context("webgpu"), Ok(Some(_))) {
                let append = || {
                    web_sys::window()?
                        .document()?
                        .body()?
                        .append_child(&web_sys::Element::from(window.canvas()))
                        .ok()
                };
                append().expect("couldn't append to document body");
            } else {
                let append = || {
                    web_sys::window()?
                        .document()?
                        .body()?
                        .set_inner_text("canvas.getContext('webgpu') returned null. Maybe your browser doesn't support webgpu, or you don't have it enabled?");
                    Some(())
                };
                append().expect("couldn't append to document body");
                return Err(());
            };
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            dx12_shader_compiler: Default::default(),
        });
        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        info!("Using adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = capabilities.formats[0];
        info!(
            "Capabilities: {:?}. Using {:?}.",
            capabilities, swapchain_format
        );

        let size = window.inner_size();

        let surface_configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![swapchain_format.add_srgb_suffix()],
        };
        surface.configure(&device, &surface_configuration);

        let staging_belt = wgpu::util::StagingBelt::new(1024);

        let interactive = SyncInteractiveKernel::create(&device, &queue, size.width, size.height);

        let buffer_blit = BufferBlit::new(
            &device,
            swapchain_format.add_srgb_suffix(),
            interactive.texture(),
            interactive.texture_size(),
            // "lol", I said, "lmao"
            !format!("{:?}", swapchain_format.add_srgb_suffix()).contains("Srgb"),
        );

        #[cfg(target_arch = "wasm32")]
        let font = wgpu_text::font::FontArc::try_from_slice(include_bytes!(
            "C:\\Windows\\Fonts\\arial.ttf"
        ))
        .unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        let font =
            wgpu_text::font::FontArc::try_from_vec(std::fs::read(find_font().unwrap()).unwrap())
                .unwrap();
        let glyph = wgpu_text::BrushBuilder::using_font(font).build_custom(
            &device,
            size.width,
            size.height,
            swapchain_format.add_srgb_suffix(),
        );

        Ok(Self {
            event_loop: Some(event_loop),
            window,
            surface,
            swapchain_format,
            device,
            queue,
            staging_belt,
            glyph,
            size,
            buffer_blit,
            fps_counter: FpsCounter::new(1.0),
            interactive,
        })
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.swapchain_format,
                width: new_size.width,
                height: new_size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![self.swapchain_format.add_srgb_suffix()],
            },
        );
        self.glyph
            .resize_view(new_size.width as f32, new_size.height as f32, &self.queue);
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
            warn!("suboptimal");
        }
        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(self.swapchain_format.add_srgb_suffix()),
            dimension: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.interactive.run(&self.device, &mut encoder);

        self.buffer_blit.set_src(
            &self.device,
            self.interactive.texture(),
            self.interactive.texture_size(),
            None,
        );
        self.buffer_blit
            .blit(&self.device, &mut encoder, &frame_view);

        self.fps_counter.tick();
        let display_text = format!(
            "{} fps\n{}",
            self.fps_counter.value(),
            self.interactive.status()
        );
        let finished_encoder = encoder.finish();

        let section = wgpu_text::section::Section::default()
            .add_text(wgpu_text::section::Text::new(&display_text));
        self.glyph.queue(section);
        self.glyph
            .process_queued(&self.device, &self.queue)
            .unwrap();
        let textbuf = self.glyph.draw(&self.device, &frame_view);

        self.staging_belt.finish();
        self.queue.submit([finished_encoder, textbuf]);
        frame.present();

        self.staging_belt.recall();

        Ok(())
    }

    pub fn run(mut self) -> ! {
        let event_loop = self.event_loop.take().unwrap();
        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == self.window.id() => {
                self.input(event);
                match event {
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        self.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        self.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => match self.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => self.resize(self.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(wgpu::SurfaceError::Timeout) => error!("Error: Timeout"),
                Err(wgpu::SurfaceError::Outdated) => error!("Error: Outdated"),
            },
            Event::MainEventsCleared => {
                #[cfg(target_arch = "wasm32")]
                {
                    // resize to fullscreen
                    use winit::dpi::PhysicalSize;
                    let window = web_sys::window().unwrap();
                    let width = window.inner_width().unwrap().as_f64().unwrap() as u32;
                    let height = window.inner_height().unwrap().as_f64().unwrap() as u32;
                    let inner_size = self.window.inner_size();
                    let new_size = PhysicalSize::new(width, height);
                    if inner_size != new_size {
                        info!("resize from {:?} to {:?}", inner_size, new_size);
                        self.window.set_inner_size(new_size);
                    }
                }

                // RedrawRequested will only trigger once, unless we manually
                // request it.
                self.window.request_redraw();
            }
            _ => {}
        });
    }
}
