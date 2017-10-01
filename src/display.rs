use glium::Surface;
use glium::glutin;
use glium::texture::texture2d;
use glium;
use std::error::Error;
use std::sync::mpsc;
use std::time::Instant;

pub enum ScreenEvent {
    Resize(u32, u32),
    KeyDown(glutin::VirtualKeyCode, Instant),
    KeyUp(glutin::VirtualKeyCode, Instant),
}

pub fn display(
    width: u32,
    height: u32,
    image_stream: mpsc::Receiver<glium::texture::RawImage2d<u8>>,
    event_send: mpsc::Sender<ScreenEvent>,
) -> Result<(), Box<Error>> {
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new();
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let display = glium::Display::new(window, context, &events_loop)?;
    let mut texture: Option<texture2d::Texture2d> = None;
    loop {
        let mut closed = false;
        events_loop.poll_events(|ev| match ev {
            glutin::Event::WindowEvent { event, .. } => {
                match event {
                    glutin::WindowEvent::Closed => closed = true,
                    glutin::WindowEvent::Resized(width, height) => {
                        match event_send.send(ScreenEvent::Resize(width, height)) {
                            Ok(()) => (),
                            Err(_) => closed = true,
                        }
                    }
                    glutin::WindowEvent::KeyboardInput {
                        input: glutin::KeyboardInput {
                            state,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                        ..
                    } => {
                        let screen_event = match state {
                            glutin::ElementState::Pressed => {
                                ScreenEvent::KeyDown(keycode, Instant::now())
                            }
                            glutin::ElementState::Released => {
                                ScreenEvent::KeyUp(keycode, Instant::now())
                            }
                        };
                        match event_send.send(screen_event) {
                            Ok(()) => (),
                            Err(_) => closed = true,
                        }
                    }
                    _ => (),
                }
            }
            _ => (),
        });
        if closed {
            return Ok(());
        }
        loop {
            let image = match image_stream.try_recv() {
                Ok(image) => image,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            };
            if texture.is_some() &&
                texture.as_ref().unwrap().dimensions() != (image.width, image.height)
            {
                texture = None;
            }
            if let Some(ref texture) = texture {
                let rect = glium::Rect {
                    left: 0,
                    bottom: 0,
                    width: image.width,
                    height: image.height,
                };
                texture.write(rect, image);
            } else {
                texture = Some(texture2d::Texture2d::new(&display, image)?);
            }
        }
        let target = display.draw();
        //target.clear_color_srgb(0.3, 0.0, 0.3, 1.0);
        if let Some(ref texture) = texture {
            let to_draw = texture.as_surface();
            to_draw.fill(&target, glium::uniforms::MagnifySamplerFilter::Nearest);
        }
        target.finish().unwrap();
    }
}
