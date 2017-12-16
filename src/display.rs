use glium::Surface;
use glium::glutin;
use glium::texture::texture2d;
use glium;
use kernel;
use std::error::Error;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

pub enum ScreenEvent {
    Resize(u32, u32),
    KeyDown(glutin::VirtualKeyCode, Instant),
    KeyUp(glutin::VirtualKeyCode, Instant),
}

struct Display {
    display: glium::Display,
    texture: Option<texture2d::Texture2d>,
    send_screen_event: mpsc::Sender<ScreenEvent>,
    recv_image: mpsc::Receiver<glium::texture::RawImage2d<'static, u8>>,
}

impl Display {
    fn keyboard_input(
        &self,
        keycode: glutin::VirtualKeyCode,
        state: glutin::ElementState,
    ) -> glutin::ControlFlow {
        match keycode {
            glutin::VirtualKeyCode::Escape => glutin::ControlFlow::Break,
            _ => {
                let screen_event = match state {
                    glutin::ElementState::Pressed => ScreenEvent::KeyDown(keycode, Instant::now()),
                    glutin::ElementState::Released => ScreenEvent::KeyUp(keycode, Instant::now()),
                };
                match self.send_screen_event.send(screen_event) {
                    Ok(()) => glutin::ControlFlow::Continue,
                    Err(_) => glutin::ControlFlow::Break,
                }
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

    fn draw(&self, target: &glium::Frame) {
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
        }
    }

    fn update_and_draw(&mut self) -> glutin::ControlFlow {
        if self.update_tex() == glutin::ControlFlow::Break {
            return glutin::ControlFlow::Break;
        }
        let target = self.display.draw();
        self.draw(&target);
        target.finish().unwrap();
        glutin::ControlFlow::Continue
    }

    fn create_window(
        events_loop: &glutin::EventsLoop,
        width: u32,
        height: u32,
    ) -> Result<glium::Display, Box<Error>> {
        let window = glutin::WindowBuilder::new()
            .with_dimensions(width, height)
            .with_title("clam5");
        let context = glutin::ContextBuilder::new().with_srgb(true);
        let display = glium::Display::new(window, context, events_loop)?;
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
        send_image: mpsc::Sender<glium::texture::RawImage2d<'static, u8>>,
        recv_screen_event: mpsc::Receiver<ScreenEvent>,
        proxy: glutin::EventsLoopProxy,
    ) {
        thread::spawn(move || {
            match kernel::interactive(width, height, &send_image, &recv_screen_event, Some(proxy)) {
                Ok(()) => (),
                Err(err) => println!("{}", err),
            }
        });
    }

    fn new(mut width: u32, mut height: u32) -> Result<(Display, glutin::EventsLoop), Box<Error>> {
        let events_loop = glutin::EventsLoop::new();
        let display = Self::create_window(&events_loop, width, height)?;
        let (send_image, recv_image) = mpsc::channel();
        let (send_screen_event, recv_screen_event) = mpsc::channel();
        Self::update_window_size(&display, &mut width, &mut height);
        Self::launch_kernel(
            width,
            height,
            send_image,
            recv_screen_event,
            events_loop.create_proxy(),
        );
        let result = Display {
            display: display,
            texture: None,
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
                } => self.keyboard_input(keycode, state),
                glutin::WindowEvent::Refresh => self.update_and_draw(),
                _ => glutin::ControlFlow::Continue,
            },
            glutin::Event::Awakened => self.update_and_draw(),
            glutin::Event::DeviceEvent { .. } => glutin::ControlFlow::Continue,
            glutin::Event::Suspended(_) => glutin::ControlFlow::Continue,
        }
    }
}

pub fn display(width: u32, height: u32) -> Result<(), Box<Error>> {
    let (mut display, mut events_loop) = Display::new(width, height)?;
    events_loop.run_forever(move |ev| display.run(ev));
    Ok(())
}
