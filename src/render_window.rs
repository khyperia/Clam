use crate::{
    SpawnLocal, buffer_blit::BufferBlit, fps_counter::FpsCounter,
    interactive::SyncInteractiveKernel,
};
use log::{error, info, warn};
use std::{cell::RefCell, rc::Rc, sync::Arc};
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{self, KeyCode, PhysicalKey},
    window::Window,
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
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
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
    let mut options = wgpu::InstanceDescriptor::new_without_display_handle();
    options.backends = wgpu::Backends::PRIMARY;
    options.flags = wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DISCARD_HAL_LABELS;
    let instance = wgpu::Instance::new(options);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .unwrap();
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .unwrap()
}

impl RenderWindow {
    pub async fn new(window: Window) -> Option<Self> {
        let window = Arc::new(window);

        #[cfg(target_arch = "wasm32")]
        {
            let err = |s| {
                web_sys::window()?.document()?.body()?.set_inner_text(s);
                Some(())
            };
            let err = |s| err(s).expect("couldn't set document body");
            let Some(canvas) = window.canvas() else {
                err("winit window.canvas() returned null");
                return None;
            };
            let Ok(Some(_)) = canvas.get_context("webgpu") else {
                err(
                    "canvas.getContext('webgpu') returned null. Maybe your browser doesn't \
                     support webgpu, or you don't have it enabled?",
                );
                return None;
            };
            let append = || {
                web_sys::window()?
                    .document()?
                    .body()?
                    .append_child(&canvas.into())
                    .ok()
            };
            append().expect("couldn't append to document body");
        }

        let mut options =
            wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window.clone()));
        options.backends = wgpu::Backends::PRIMARY;
        options.flags = wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DISCARD_HAL_LABELS;
        let instance = wgpu::Instance::new(options);
        let size: winit::dpi::PhysicalSize<u32> = window.inner_size();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .unwrap();

        info!("Using adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = capabilities.formats[0];
        info!("Capabilities: {capabilities:?}. Using {swapchain_format:?}.");

        let surface_configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![swapchain_format.add_srgb_suffix()],
        };
        surface.configure(&device, &surface_configuration);

        let staging_belt = wgpu::util::StagingBelt::new(device.clone(), 1024);

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
        let font = wgpu_text::glyph_brush::ab_glyph::FontArc::try_from_slice(include_bytes!(
            "/usr/share/fonts/TTF/FiraMono-Regular.ttf"
        ))
        .unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        let font = wgpu_text::glyph_brush::ab_glyph::FontArc::try_from_vec(
            std::fs::read(find_font().unwrap()).unwrap(),
        )
        .unwrap();
        let glyph = wgpu_text::BrushBuilder::using_font(font).build(
            &device,
            size.width,
            size.height,
            swapchain_format.add_srgb_suffix(),
        );

        Some(Self {
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
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: new_size.width,
                height: new_size.height,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
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
            event:
                KeyEvent {
                    state,
                    physical_key: keyboard::PhysicalKey::Code(key),
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

    fn render(&mut self, frame: wgpu::SurfaceTexture) {
        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(self.swapchain_format.add_srgb_suffix()),
            dimension: None,
            usage: None,
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

        let section = wgpu_text::glyph_brush::Section::default()
            .add_text(wgpu_text::glyph_brush::Text::new(&display_text));
        self.glyph
            .queue(&self.device, &self.queue, vec![section])
            .unwrap();
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.glyph.draw(&mut rpass);
        }

        let finished_encoder = encoder.finish();

        self.staging_belt.finish();
        self.queue.submit([finished_encoder]);
        self.queue.present(frame);

        self.staging_belt.recall();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if window_id == self.window.id() {
            self.input(&event);
            match event {
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                    ..
                } => event_loop.exit(),
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(physical_size) => {
                    self.resize(physical_size);
                }
                // WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                //     self.resize(**new_inner_size);
                // }
                WindowEvent::RedrawRequested => match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(surface_texture) => {
                        self.render(surface_texture)
                    }
                    wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                        warn!("suboptimal");
                        self.render(surface_texture)
                    }
                    wgpu::CurrentSurfaceTexture::Timeout => error!("Error: Timeout"),
                    wgpu::CurrentSurfaceTexture::Occluded => error!("Error: Occluded"),
                    wgpu::CurrentSurfaceTexture::Outdated => error!("Error: Outdated"),
                    wgpu::CurrentSurfaceTexture::Lost => error!("Error: Lost"),
                    wgpu::CurrentSurfaceTexture::Validation => error!("Error: Validation"),
                },
                _ => {}
            }
        }
    }
}

pub fn run(spawn_local: SpawnLocal) {
    EventLoop::new()
        .unwrap()
        .run_app(&mut AppHandler {
            spawn_local,
            render_window: Rc::new(RefCell::new(None)),
        })
        .unwrap();
}

struct AppHandler {
    spawn_local: SpawnLocal,
    render_window: Rc<RefCell<Option<RenderWindow>>>,
}

impl ApplicationHandler<()> for AppHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.render_window.borrow().is_none() {
            let window = event_loop.create_window(Default::default()).unwrap();
            let render_window = RenderWindow::new(window);
            let r = self.render_window.clone();
            (self.spawn_local)(Box::pin(async move {
                *r.borrow_mut() = render_window.await;
            }));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(render_window) = &mut *self.render_window.borrow_mut() {
            render_window.window_event(event_loop, window_id, event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(render_window) = &mut *self.render_window.borrow_mut() {
            #[cfg(target_arch = "wasm32")]
            {
                // resize to fullscreen
                use winit::dpi::PhysicalSize;
                let window = web_sys::window().unwrap();
                let width = window.inner_width().unwrap().as_f64().unwrap() as u32;
                let height = window.inner_height().unwrap().as_f64().unwrap() as u32;
                let inner_size = render_window.window.inner_size();
                let new_size = PhysicalSize::new(width, height);
                if inner_size != new_size {
                    info!("resize from {:?} to {:?}", inner_size, new_size);
                    match render_window.window.request_inner_size(new_size) {
                        None => (),
                        Some(physical_size) => render_window.resize(physical_size),
                    }
                }
            }

            // RedrawRequested will only trigger once, unless we manually request it.
            render_window.window.request_redraw();
        }
    }
}
