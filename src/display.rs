use glium::Surface;
use glium::glutin;
use glium::texture::texture2d;
use glium;
use glium_text_rusttype;
use input;
use kernel;
use settings;
use failure;
use failure::Error;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

pub enum ScreenEvent {
    Resize(u32, u32),
}

const FONT_SIZE: u32 = 24;

struct ClamDisplay {
    display: glium::Display,
    texture: Option<texture2d::Texture2d>,
    text_system: glium_text_rusttype::TextSystem,
    font_texture: glium_text_rusttype::FontTexture,
    settings_input: Arc<Mutex<(settings::Settings, input::Input)>>,
    send_screen_event: mpsc::Sender<ScreenEvent>,
    recv_image: mpsc::Receiver<glium::texture::RawImage2d<'static, u8>>,
}

fn find_font() -> Result<::std::fs::File, Error> {
    let locations = [
        "/usr/share/fonts/TTF/FiraMono-Regular.ttf",
        "/usr/share/fonts/TTF/FiraSans-Regular.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
    ];
    for location in &locations {
        match ::std::fs::File::open(location) {
            Ok(file) => return Ok(file),
            Err(_) => (),
        }
    }
    return Err(failure::err_msg("No font found"));
}

impl ClamDisplay {
    fn keyboard_input(
        &self,
        keycode: glutin::VirtualKeyCode,
        state: glutin::ElementState,
    ) -> glutin::ControlFlow {
        match keycode {
            glutin::VirtualKeyCode::Escape => glutin::ControlFlow::Break,
            _ => {
                let mut locked = self.settings_input.lock().unwrap();
                let (ref mut settings, ref mut input) = *locked;
                match state {
                    glutin::ElementState::Pressed => input.key_down(keycode, Instant::now(), settings),
                    glutin::ElementState::Released => input.key_up(keycode, Instant::now(), settings),
                }
                glutin::ControlFlow::Continue
            }
        }
    }

    fn update_tex(&mut self) -> glutin::ControlFlow {
        loop {
            let image = match self.recv_image.try_recv() {
                Ok(image) => image,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return glutin::ControlFlow::Break,
            };
            if self.texture.is_some()
                && self.texture.as_ref().unwrap().dimensions() != (image.width, image.height)
            {
                self.texture = None;
            }
            if let Some(ref texture) = self.texture {
                let rect = glium::Rect {
                    left: 0,
                    bottom: 0,
                    width: image.width,
                    height: image.height,
                };
                texture.write(rect, image);
            } else {
                self.texture = Some(
                    texture2d::Texture2d::new(&self.display, image)
                        .expect("Failed to create texture2d"),
                );
            }
        }
        glutin::ControlFlow::Continue
    }

    fn get_status(&self) -> String {
        let mut locked = self.settings_input.lock().unwrap();
        let (ref mut settings, ref mut input) = *locked;
        input.integrate(settings);
        settings::settings_status(settings, input)
    }

    fn draw_text(&self, target: &mut glium::Frame, width: u32, height: u32) {
        let status = self.get_status();
        let mut text = glium_text_rusttype::TextDisplay::new(
            &self.text_system,
            &self.font_texture,
            &status,
        );
        let mut text_height = 0.0;
        for line in status.lines() {
            text.set_text(line);
            text_height += text.get_height();
            let upscale = 2;
            let x_scale = (upscale * FONT_SIZE) as f32 / width as f32;
            let y_scale = (upscale * FONT_SIZE) as f32 / height as f32;

            let matrix = [
                [x_scale, 0.0, 0.0, 0.0],
                [0.0, y_scale, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, 1.0 - text_height * y_scale, 0.0, 1.0],
            ];
            glium_text_rusttype::draw(
                &text,
                &self.text_system,
                target,
                matrix,
                (1.0, 0.5, 0.0, 1.0),
            ).unwrap();
        }
    }

    fn draw(&self, target: &mut glium::Frame) {
        if let Some(ref texture) = self.texture {
            let to_draw = texture.as_surface();
            let (width, height) = to_draw.get_dimensions();
            let blit_target = glium::BlitTarget {
                left: 0,
                bottom: height,
                width: width as i32,
                height: -(height as i32),
            };
            to_draw.blit_whole_color_to(
                target,
                &blit_target,
                glium::uniforms::MagnifySamplerFilter::Nearest,
            );
            self.draw_text(target, width, height);
        }
    }

    fn update_and_draw(&mut self) -> glutin::ControlFlow {
        if self.update_tex() == glutin::ControlFlow::Break {
            return glutin::ControlFlow::Break;
        }
        let mut target = self.display.draw();
        self.draw(&mut target);
        target.finish().unwrap();
        glutin::ControlFlow::Continue
    }

    fn create_window(
        events_loop: &glutin::EventsLoop,
        width: u32,
        height: u32,
    ) -> Result<glium::Display, Error> {
        let window = glutin::WindowBuilder::new()
            .with_dimensions(width, height)
            .with_title("clam5");
        let context = glutin::ContextBuilder::new();
        let display = glium::Display::new(window, context, events_loop).unwrap(); // TODO: don't unwrap
        Ok(display)
    }

    fn update_window_size(display: &glium::Display, width: &mut u32, height: &mut u32) {
        if let Some((new_width, new_height)) = display.gl_window().get_inner_size() {
            // On HiDPI screens, this might be different than what was passed in
            *width = new_width;
            *height = new_height;
        }
    }

    fn launch_kernel(
        width: u32,
        height: u32,
        settings_input: Arc<Mutex<(settings::Settings, input::Input)>>,
        send_image: mpsc::Sender<glium::texture::RawImage2d<'static, u8>>,
        recv_screen_event: mpsc::Receiver<ScreenEvent>,
        proxy: glutin::EventsLoopProxy,
    ) {
        thread::spawn(move || match kernel::interactive(
            width,
            height,
            settings_input,
            &send_image,
            &recv_screen_event,
            Some(proxy),
        ) {
            Ok(()) => (),
            Err(err) => println!("{}", err),
        });
    }

    fn new(mut width: u32, mut height: u32) -> Result<(ClamDisplay, glutin::EventsLoop), Error> {
        let events_loop = glutin::EventsLoop::new();
        let display = Self::create_window(&events_loop, width, height)?;
        let text_system = glium_text_rusttype::TextSystem::new(&display);
        let font_texture = glium_text_rusttype::FontTexture::new(
            &display,
            find_font()?,
            FONT_SIZE,
            glium_text_rusttype::FontTexture::ascii_character_list(),
        ).unwrap();
        let settings_input = Arc::new(Mutex::new((settings::init_settings(), input::Input::new())));
        let (send_image, recv_image) = mpsc::channel();
        let (send_screen_event, recv_screen_event) = mpsc::channel();
        Self::update_window_size(&display, &mut width, &mut height);
        Self::launch_kernel(
            width,
            height,
            settings_input.clone(),
            send_image,
            recv_screen_event,
            events_loop.create_proxy(),
        );
        let result = ClamDisplay {
            display: display,
            texture: None,
            text_system: text_system,
            font_texture: font_texture,
            settings_input: settings_input,
            send_screen_event: send_screen_event,
            recv_image: recv_image,
        };
        Ok((result, events_loop))
    }

    fn run(&mut self, event: glutin::Event) -> glutin::ControlFlow {
        match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::Closed => glutin::ControlFlow::Break,
                glutin::WindowEvent::Resized(width, height) => match self.send_screen_event
                    .send(ScreenEvent::Resize(width, height))
                {
                    Ok(()) => glutin::ControlFlow::Continue,
                    Err(_) => glutin::ControlFlow::Break,
                },
                glutin::WindowEvent::KeyboardInput {
                    input:
                        glutin::KeyboardInput {
                            state,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                } => {
                    if self.keyboard_input(keycode, state) == glutin::ControlFlow::Break {
                        return glutin::ControlFlow::Break;
                    }
                    self.update_and_draw()
                },
                glutin::WindowEvent::Refresh => self.update_and_draw(),
                _ => glutin::ControlFlow::Continue,
            },
            glutin::Event::Awakened => self.update_and_draw(),
            glutin::Event::DeviceEvent { .. } => glutin::ControlFlow::Continue,
            glutin::Event::Suspended(_) => glutin::ControlFlow::Continue,
        }
    }
}

pub fn display(width: u32, height: u32) -> Result<(), Error> {
    let (mut display, mut events_loop) = ClamDisplay::new(width, height)?;
    events_loop.run_forever(move |ev| display.run(ev));
    Ok(())
}
