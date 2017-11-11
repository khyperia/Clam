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

fn keyboard_input(
    keycode: glutin::VirtualKeyCode,
    state: glutin::ElementState,
    event_send: &mpsc::Sender<ScreenEvent>,
) -> glutin::ControlFlow {
    match keycode {
        glutin::VirtualKeyCode::Escape => glutin::ControlFlow::Break,
        _ => {
            let screen_event = match state {
                glutin::ElementState::Pressed => ScreenEvent::KeyDown(keycode, Instant::now()),
                glutin::ElementState::Released => ScreenEvent::KeyUp(keycode, Instant::now()),
            };
            match event_send.send(screen_event) {
                Ok(()) => glutin::ControlFlow::Continue,
                Err(_) => glutin::ControlFlow::Break,
            }
        }
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

fn update_and_draw(
    texture: &mut Option<texture2d::Texture2d>,
    display: &glium::Display,
    recv_image: &mpsc::Receiver<glium::texture::RawImage2d<u8>>,
) -> glutin::ControlFlow {
    if update_tex(texture, display, recv_image) {
        return glutin::ControlFlow::Break;
    }
    let target = display.draw();
    if let Some(ref texture) = *texture {
        draw(&target, texture);
    }
    target.finish().unwrap();
    glutin::ControlFlow::Continue
}

pub fn display(mut width: u32, mut height: u32) -> Result<(), Box<Error>> {
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_dimensions(width, height)
        .with_title("clam5");
    let context = glutin::ContextBuilder::new().with_srgb(true);
    let display = glium::Display::new(window, context, &events_loop)?;
    let mut texture: Option<texture2d::Texture2d> = None;
    if let Some((new_width, new_height)) = display.gl_window().get_inner_size_pixels() {
        // On HiDPI screens, this might be different than what was passed in
        width = new_width;
        height = new_height;
    }
    let proxy = events_loop.create_proxy();
    let (send_image, recv_image) = mpsc::channel();
    let (send_screen_event, recv_screen_event) = mpsc::channel();
    thread::spawn(move || match kernel::interactive(
        width,
        height,
        &send_image,
        &recv_screen_event,
        Some(proxy),
    ) {
        Ok(()) => (),
        Err(err) => println!("{}", err),
    });
    events_loop.run_forever(move |ev| match ev {
        glutin::Event::WindowEvent { event, .. } => {
            match event {
                glutin::WindowEvent::Closed => glutin::ControlFlow::Break,
                glutin::WindowEvent::Resized(width, height) => {
                    match send_screen_event.send(ScreenEvent::Resize(width, height)) {
                        Ok(()) => glutin::ControlFlow::Continue,
                        Err(_) => glutin::ControlFlow::Break,
                    }
                }
                glutin::WindowEvent::KeyboardInput {
                    input: glutin::KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                    ..
                } => keyboard_input(keycode, state, &send_screen_event),
                glutin::WindowEvent::Refresh => {
                    update_and_draw(&mut texture, &display, &recv_image)
                }
                _ => glutin::ControlFlow::Continue,
            }
        }
        glutin::Event::Awakened => update_and_draw(&mut texture, &display, &recv_image),
        glutin::Event::DeviceEvent { .. } => glutin::ControlFlow::Continue,
        glutin::Event::Suspended(_) => glutin::ControlFlow::Continue,
    });
    Ok(())
}
