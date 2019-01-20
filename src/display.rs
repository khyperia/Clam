use failure::err_msg;
use failure::Error;
use fps_counter::FpsCounter;
use input;
use interactive::{DownloadResult, InteractiveKernel};
use sdl2::event::Event;
use sdl2::event::WindowEvent;
use sdl2::init;
use sdl2::pixels::Color;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::Canvas;
use sdl2::render::Texture;
use sdl2::render::TextureCreator;
use sdl2::ttf;
use sdl2::ttf::Font;
use sdl2::video::Window;
use sdl2::video::WindowContext;
use settings::Settings;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct Image {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
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

pub enum ScreenEvent {
    Resize(u32, u32),
    KernelChanged,
}

fn find_font() -> Result<&'static Path, Error> {
    let locations: [&'static Path; 6] = [
        "/usr/share/fonts/TTF/FiraMono-Regular.ttf".as_ref(),
        "/usr/share/fonts/TTF/FiraSans-Regular.ttf".as_ref(),
        "C:\\Windows\\Fonts\\arial.ttf".as_ref(),
        "/usr/share/fonts/TTF/DejaVuSans.ttf".as_ref(),
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf".as_ref(),
        "/Library/Fonts/Andale Mono.ttf".as_ref(),
    ];
    for &location in &locations {
        if location.exists() {
            return Ok(location);
        }
    }
    Err(err_msg("No font found"))
}

fn render_text_one(
    font: &ttf::Font,
    creator: &TextureCreator<WindowContext>,
    canvas: &mut Canvas<Window>,
    text: &str,
    color: Color,
    offset_x: i32,
    offset_y: i32,
) -> Result<(), Error> {
    let spacing = font.recommended_line_spacing();
    for (line_index, line) in text.lines().enumerate() {
        let rendered = font.render(line).solid(color)?;
        let width = rendered.width();
        let height = rendered.height();
        let tex = creator.create_texture_from_surface(rendered)?;
        let y = 10 + offset_y + line_index as i32 * spacing;
        canvas
            .copy(&tex, None, Rect::new(10 + offset_x, y, width, height))
            .expect("Could not display text");
    }
    Ok(())
}

fn render_text(
    font: &ttf::Font,
    settings_input: &Arc<Mutex<(Settings, input::Input)>>,
    creator: &TextureCreator<WindowContext>,
    canvas: &mut Canvas<Window>,
    render_fps: f64,
    window_fps: f64,
) -> Result<(), Error> {
    let fps_line_1 = format!("{:.2} render fps", render_fps);
    let fps_line_2 = format!("{:.2} window fps", window_fps);
    let mut locked = settings_input.lock().unwrap();
    let (ref mut settings, ref mut input) = *locked;
    input.integrate(settings);
    let text = format!("{}\n{}\n{}", fps_line_1, fps_line_2, settings.status(input));
    render_text_one(font, creator, canvas, &text, Color::RGB(0, 0, 0), 1, 1)?;
    render_text_one(
        font,
        creator,
        canvas,
        &text,
        Color::RGB(255, 192, 192),
        0,
        0,
    )?;
    Ok(())
}

struct WindowData {
    image_width: u32,
    image_height: u32,
    render_fps: FpsCounter,
    window_fps: FpsCounter,
}

fn draw<'a>(
    canvas: &mut Canvas<Window>,
    creator: &'a TextureCreator<WindowContext>,
    font: &Font,
    interactive_kernel: &InteractiveKernel,
    settings_input: &Arc<Mutex<(Settings, input::Input)>>,
    texture: &mut Texture<'a>,
    window_data: &mut WindowData,
) -> Result<bool, Error> {
    match interactive_kernel.download() {
        DownloadResult::NoMoreImages => return Ok(false),
        DownloadResult::NoneAtPresent => (),
        DownloadResult::Image(image) => {
            window_data.render_fps.tick();
            if window_data.image_width != image.width || window_data.image_height != image.height {
                window_data.image_width = image.width;
                window_data.image_height = image.height;
                *texture = creator.create_texture_streaming(
                    PixelFormatEnum::ABGR8888,
                    window_data.image_width,
                    window_data.image_height,
                )?;
            }

            texture.update(None, &image.data, image.width as usize * 4)?;
        }
    }
    // let image = match image_stream.try_recv() {
    //     Ok(image) => Some(image),
    //     Err(mpsc::TryRecvError::Empty) => None,
    //     Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
    // };

    //if let Some(image) = image {
    //}

    window_data.window_fps.tick();

    let (output_width, output_height) = canvas.output_size().map_err(failure::err_msg)?;
    let src = Rect::new(0, 0, window_data.image_width, window_data.image_height);
    let dest = Rect::new(0, 0, output_width, output_height);
    canvas
        .copy(texture, src, dest)
        .expect("Could not display image");
    render_text(
        font,
        settings_input,
        creator,
        canvas,
        window_data.render_fps.value(),
        window_data.window_fps.value(),
    )?;
    canvas.present();
    Ok(true)
}

pub fn display(width: u32, height: u32) -> Result<(), Error> {
    let sdl = init().expect("SDL failed to init");
    let video = sdl.video().expect("SDL does not have video");
    let mut event_pump = sdl.event_pump().expect("SDL does not have event pump");

    let window = video.window("Scopie", width, height).resizable().build()?;
    let mut canvas = window.into_canvas().present_vsync().build()?;
    let creator = canvas.texture_creator();
    let mut texture = creator.create_texture_streaming(PixelFormatEnum::ABGR8888, width, height)?;

    let ttf = ttf::init()?;
    let font = ttf.load_font(find_font()?, 20).expect("Cannot open font");

    let settings_input = Arc::new(Mutex::new((Settings::new(), input::Input::new())));

    let interactive_kernel = InteractiveKernel::create(width, height, settings_input.clone())?;

    let mut window_data = WindowData {
        image_width: width,
        image_height: height,
        render_fps: FpsCounter::new(1.0),
        window_fps: FpsCounter::new(1.0),
    };

    loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Window {
                    win_event: WindowEvent::Resized(width, height),
                    ..
                } if width > 0 && height > 0 => {
                    interactive_kernel.resize(width as u32, height as u32);
                    // match event_stream.send(ScreenEvent::Resize(width as u32, height as u32)) {
                    //     Ok(()) => (),
                    //     Err(_) => return Ok(()),
                    // }
                }
                Event::KeyDown {
                    scancode: Some(scancode),
                    ..
                } => {
                    let mut locked = settings_input.lock().unwrap();
                    let (ref mut settings, ref mut input) = *locked;
                    input.key_down(scancode, settings);
                }
                Event::KeyUp {
                    scancode: Some(scancode),
                    ..
                } => {
                    let mut locked = settings_input.lock().unwrap();
                    let (ref mut settings, ref mut input) = *locked;
                    input.key_up(scancode, settings);
                }
                Event::Quit { .. } => return Ok(()),
                _ => (),
            }
        }
        if !draw(
            &mut canvas,
            &creator,
            &font,
            &interactive_kernel,
            &settings_input,
            &mut texture,
            &mut window_data,
        )? {
            return Ok(());
        }
    }
}
