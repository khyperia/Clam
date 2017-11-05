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

// returns true if should close
fn handle_event(ev: glutin::Event, event_send: &mpsc::Sender<ScreenEvent>) -> bool {
    if let glutin::Event::WindowEvent { event, .. } = ev {
        match event {
            glutin::WindowEvent::Closed => true,
            glutin::WindowEvent::Resized(width, height) => {
                match event_send.send(ScreenEvent::Resize(width, height)) {
                    Ok(()) => false,
                    Err(_) => true,
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
                match keycode {
                    glutin::VirtualKeyCode::Escape => true,
                    _ => {
                        let screen_event = match state {
                            glutin::ElementState::Pressed => {
                                ScreenEvent::KeyDown(keycode, Instant::now())
                            }
                            glutin::ElementState::Released => {
                                ScreenEvent::KeyUp(keycode, Instant::now())
                            }
                        };
                        match event_send.send(screen_event) {
                            Ok(()) => false,
                            Err(_) => true,
                        }
                    }
                }
            }
            _ => false,
        }
    } else {
        false
    }
}

// returns true if should close
fn update_tex(
    texture: &mut Option<texture2d::Texture2d>,
    display: &glium::Display,
    image_stream: &mpsc::Receiver<glium::texture::RawImage2d<u8>>,
) -> bool {
    loop {
        let image = match image_stream.try_recv() {
            Ok(image) => image,
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => return true,
        };
        if texture.is_some() &&
            texture.as_ref().unwrap().dimensions() != (image.width, image.height)
        {
            *texture = None;
        }
        if let &mut Some(ref texture) = texture {
            let rect = glium::Rect {
                left: 0,
                bottom: 0,
                width: image.width,
                height: image.height,
            };
            texture.write(rect, image);
        } else {
            *texture = Some(texture2d::Texture2d::new(display, image).expect(
                "Failed to create texture2d",
            ));
        }
    }
    false
}

fn draw(target: &glium::Frame, texture: &texture2d::Texture2d) {
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

pub fn display(
    width: u32,
    height: u32,
    image_stream: &mpsc::Receiver<glium::texture::RawImage2d<u8>>,
    event_send: &mpsc::Sender<ScreenEvent>,
) -> Result<(), Box<Error>> {
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_dimensions(width, height)
        .with_title("clam5");
    let context = glutin::ContextBuilder::new().with_srgb(true);
    let display = glium::Display::new(window, context, &events_loop)?;
    let mut texture: Option<texture2d::Texture2d> = None;
    if let Some((new_width, new_height)) = display.gl_window().get_inner_size_pixels() {
        // On HiDPI screens, this might be different than what was passed in
        if new_width != width || new_height != height {
            event_send.send(ScreenEvent::Resize(new_width, new_height))?;
        }
    }
    loop {
        let mut closed = false;
        events_loop.poll_events(|ev| closed = handle_event(ev, event_send));
        if closed {
            return Ok(());
        }
        if update_tex(&mut texture, &display, image_stream) {
            return Ok(());
        }
        let target = display.draw();
        //target.clear_color_srgb(0.3, 0.0, 0.3, 1.0);
        if let Some(ref texture) = texture {
            draw(&target, texture);
        }
        target.finish().unwrap();
    }
}
