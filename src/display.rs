use failure::err_msg;
use failure::Error;
use input;
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

type KernelFn = Fn(
        u32,
        u32,
        &Arc<Mutex<(Settings, input::Input)>>,
        &mpsc::SyncSender<Image>,
        &mpsc::Receiver<ScreenEvent>,
    ) -> Result<(), Error>
    + Sync;

fn launch_kernel(
    width: u32,
    height: u32,
    settings_input: Arc<Mutex<(Settings, input::Input)>>,
    send_image: mpsc::SyncSender<Image>,
    recv_screen_event: mpsc::Receiver<ScreenEvent>,
    kernel_fn: &'static KernelFn,
) {
    thread::spawn(move || {
        match kernel_fn(
            width,
            height,
            &settings_input,
            &send_image,
            &recv_screen_event,
        ) {
            Ok(()) => (),
            Err(err) => println!("Error in kernel thread: {}", err),
        }
    });
}

pub fn find_font() -> Result<&'static Path, Error> {
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

struct FpsCounter {
    weight: f64,
    last_fps: Instant,
    spf: f64,
}

impl FpsCounter {
    fn new(weight: f64) -> Self {
        Self {
            weight,
            last_fps: Instant::now(),
            spf: 1.0,
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        let duration = now.duration_since(self.last_fps);
        self.last_fps = now;

        // as_secs returns u64, subsec_nanos returns u32
        let time = duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / 1_000_000_000.0;

        let weight = self.weight / self.spf;
        self.spf = (time + (self.spf * weight)) / (weight + 1.0);
    }

    fn value(&self) -> f64 {
        1.0 / self.spf
    }
}

struct WindowData {
    width: u32,
    height: u32,
    render_fps: FpsCounter,
    window_fps: FpsCounter,
}

fn draw<'a>(
    canvas: &mut Canvas<Window>,
    creator: &'a TextureCreator<WindowContext>,
    font: &Font,
    image_stream: &mpsc::Receiver<Image>,
    settings_input: &Arc<Mutex<(Settings, input::Input)>>,
    texture: &mut Texture<'a>,
    window_data: &mut WindowData,
) -> Result<(), Error> {
    let image = match image_stream.try_recv() {
        Ok(image) => Some(image),
        Err(mpsc::TryRecvError::Empty) => None,
        Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
    };

    if let Some(image) = image {
        window_data.render_fps.tick();
        if window_data.width != image.width || window_data.height != image.height {
            window_data.width = image.width;
            window_data.height = image.height;
            *texture = creator.create_texture_streaming(
                PixelFormatEnum::ABGR8888,
                window_data.width,
                window_data.height,
            )?;
        }

        texture.update(None, &image.data, image.width as usize * 4)?;
    }

    window_data.window_fps.tick();

    let rect = Rect::new(0, 0, window_data.width, window_data.height);
    canvas
        .copy(texture, rect, rect)
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
    Ok(())
}

pub fn display(width: u32, height: u32, kernel_fn: &'static KernelFn) -> Result<(), Error> {
    let sdl = init().expect("SDL failed to init");
    let video = sdl.video().expect("SDL does not have video");
    let mut event_pump = sdl.event_pump().expect("SDL does not have event pump");

    let window = video.window("Scopie", width, height).resizable().build()?;
    let mut canvas = window.into_canvas().present_vsync().build()?;
    let creator = canvas.texture_creator();
    let mut texture = creator.create_texture_streaming(PixelFormatEnum::ABGR8888, width, height)?;

    let ttf = ttf::init()?;
    let font = ttf.load_font(find_font()?, 20).expect("Cannot open font");

    let (send_image, image_stream) = mpsc::sync_channel(2);
    let (event_stream, recv_screen_event) = mpsc::channel();
    let settings_input = Arc::new(Mutex::new((Settings::new(), input::Input::new())));
    launch_kernel(
        width,
        height,
        settings_input.clone(),
        send_image,
        recv_screen_event,
        kernel_fn,
    );

    let mut window_data = WindowData {
        width,
        height,
        render_fps: FpsCounter::new(1.0),
        window_fps: FpsCounter::new(1.0),
    };

    let mut draw = || {
        draw(
            &mut canvas,
            &creator,
            &font,
            &image_stream,
            &settings_input,
            &mut texture,
            &mut window_data,
        )
    };

    loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Window {
                    win_event: WindowEvent::Resized(width, height),
                    ..
                }
                    if width > 0 && height > 0 =>
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
                    let now = Instant::now();
                    {
                        let mut locked = settings_input.lock().unwrap();
                        let (ref mut settings, ref mut input) = *locked;
                        input.key_down(scancode, now, settings);
                    }
                }
                Event::KeyUp {
                    scancode: Some(scancode),
                    ..
                } => {
                    let now = Instant::now();
                    {
                        let mut locked = settings_input.lock().unwrap();
                        let (ref mut settings, ref mut input) = *locked;
                        input.key_up(scancode, now, settings);
                    }
                }
                Event::Quit { .. } => return Ok(()),
                _ => (),
            }
        }
        draw()?;
    }
}
