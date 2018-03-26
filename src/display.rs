use sdl2::video::Window;
use sdl2::video::WindowContext;
use failure::Error;
use failure::err_msg;
use input;
use kernel;
use sdl2::event::Event;
use sdl2::event::WindowEvent;
use sdl2::init;
use sdl2::pixels::Color;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::Canvas;
use sdl2::render::TextureCreator;
use sdl2::ttf;
use settings;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

pub struct Image {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub enum ScreenEvent {
    Resize(u32, u32),
}

impl Image {
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Image {
        Image {
            data,
            width,
            height,
        }
    }
}

fn launch_kernel(
    width: u32,
    height: u32,
    settings_input: Arc<Mutex<(settings::Settings, input::Input)>>,
    send_image: mpsc::Sender<Image>,
    recv_screen_event: mpsc::Receiver<ScreenEvent>,
) {
    thread::spawn(move || {
        match kernel::interactive(
            width,
            height,
            &settings_input,
            &send_image,
            &recv_screen_event,
        ) {
            Ok(()) => (),
            Err(err) => println!("{}", err),
        }
    });
}

pub fn find_font() -> Result<&'static Path, Error> {
    let locations: [&'static Path; 5] = [
        "/usr/share/fonts/TTF/FiraMono-Regular.ttf".as_ref(),
        "/usr/share/fonts/TTF/FiraSans-Regular.ttf".as_ref(),
        "C:\\Windows\\Fonts\\arial.ttf".as_ref(),
        "/usr/share/fonts/TTF/DejaVuSans.ttf".as_ref(),
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf".as_ref(),
    ];
    for &location in &locations {
        if location.exists() {
            return Ok(location);
        }
    }
    return Err(err_msg("No font found"));
}

fn render_text(
    font: &ttf::Font,
    settings_input: &Arc<Mutex<(settings::Settings, input::Input)>>,
    creator: &TextureCreator<WindowContext>,
    canvas: &mut Canvas<Window>,
) -> Result<(), Error> {
    let mut locked = settings_input.lock().unwrap();
    let (ref mut settings, ref mut input) = *locked;
    input.integrate(settings);
    let text = settings::settings_status(settings, input);
    let spacing = font.recommended_line_spacing();
    for (line_index, line) in text.lines().enumerate() {
        let rendered = font.render(&line).solid(Color::RGB(255, 64, 64))?;
        let width = rendered.width();
        let height = rendered.height();
        let tex = creator.create_texture_from_surface(rendered)?;
        let y = 10 + line_index as i32 * spacing;
        canvas
            .copy(&tex, None, Rect::new(10, y, width, height))
            .expect("Could not display text");
    }
    Ok(())
}

pub fn display(mut width: u32, mut height: u32) -> Result<(), Error> {
    let sdl = init().expect("SDL failed to init");
    let video = sdl.video().expect("SDL does not have video");
    let window = video.window("Scopie", width, height).resizable().build()?;
    let mut canvas = window.into_canvas().present_vsync().build()?;
    let creator = canvas.texture_creator();
    let mut texture = creator.create_texture_streaming(PixelFormatEnum::RGBX8888, width, height)?;
    let mut event_pump = sdl.event_pump().expect("SDL doesn't have event pump");
    let ttf = ttf::init()?;
    let font = ttf.load_font(find_font()?, 20).expect("Cannot open font");

    let (send_image, image_stream) = mpsc::channel();
    let (event_stream, recv_screen_event) = mpsc::channel();
    let settings_input = Arc::new(Mutex::new((settings::init_settings(), input::Input::new())));
    launch_kernel(
        width,
        height,
        settings_input.clone(),
        send_image,
        recv_screen_event,
    );

    loop {
        while let Some(event) = event_pump.poll_event() {
            match event {
                Event::Window {
                    win_event: WindowEvent::Resized(width, height),
                    ..
                } if width > 0 && height > 0 =>
                {
                    match event_stream.send(ScreenEvent::Resize(width as u32, height as u32)) {
                        Ok(()) => (),
                        Err(_) => return Ok(()),
                    }
                }
                Event::KeyDown {
                    scancode: Some(scancode),
                    ..
                } => {
                    let mut locked = settings_input.lock().unwrap();
                    let (ref mut settings, ref mut input) = *locked;
                    input.key_down(scancode, Instant::now(), settings);
                }
                Event::KeyUp {
                    scancode: Some(scancode),
                    ..
                } => {
                    let mut locked = settings_input.lock().unwrap();
                    let (ref mut settings, ref mut input) = *locked;
                    input.key_up(scancode, Instant::now(), settings);
                }
                Event::Quit { .. } => return Ok(()),
                _ => (),
            }
        }

        loop {
            let image = match image_stream.try_recv() {
                Ok(image) => image,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            };
            if width != image.width || height != image.height {
                width = image.width;
                height = image.height;
                texture = creator.create_texture_streaming(None, width, height)?;
            }
            texture.update(None, &image.data, image.width as usize * 4)?;
            //println!("frame");
        }

        let rect = Rect::new(0, 0, width, height);
        canvas
            .copy(&texture, rect, rect)
            .expect("Could not display image");
        render_text(&font, &settings_input, &creator, &mut canvas)?;
        canvas.present();
    }
}
