extern crate failure;
extern crate ocl;
extern crate png;
extern crate sdl2;

mod display;
mod input;
mod kernel;
mod mandelbox_cfg;
mod progress;
mod settings;

use display::Image;
use display::ScreenEvent;
use failure::Error;
use input::Input;
use kernel::Kernel;
use progress::Progress;
use sdl2::EventSubsystem;
use settings::KeyframeList;
use settings::Settings;
use std::env::args;
use std::sync::{mpsc, Arc, Mutex};

pub fn interactive(
    width: u32,
    height: u32,
    settings_input: &Arc<Mutex<(Settings, Input)>>,
    send_image: &mpsc::Sender<Image>,
    screen_events: &mpsc::Receiver<ScreenEvent>,
    event_system: &EventSubsystem,
) -> Result<(), Error> {
    let mut kernel = Kernel::new(width, height, &settings_input.lock().unwrap().0)?;
    event_system
        .register_custom_event::<()>()
        .expect("Failed to register custom event");
    loop {
        loop {
            let event = match screen_events.try_recv() {
                Ok(event) => event,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            };

            match event {
                ScreenEvent::Resize(width, height) => kernel.resize(width, height)?,
            }
        }

        let settings = {
            let mut locked = settings_input.lock().unwrap();
            let (ref mut settings, ref mut input) = *locked;
            input.integrate(settings);
            if settings.check_rebuild() {
                kernel.rebuild_self(settings)?;
            }
            (*settings).clone()
        };
        let image = kernel.run(&settings, true)?.unwrap();
        match send_image.send(image) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        };
        match event_system.push_custom_event(()) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        }
    }
}

fn save_image(image: &Image, path: &str) -> Result<(), Error> {
    use png::HasParameters;
    let file = ::std::fs::File::create(path)?;
    let w = &mut ::std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, image.width, image.height);
    encoder.set(png::ColorType::RGBA).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&image.data)?;
    Ok(())
}

pub fn headless(width: u32, height: u32, rpp: u32) -> Result<(), Error> {
    let mut settings = Settings::new();
    settings.load("settings.clam5")?;
    settings.all_constants();
    let mut kernel = Kernel::new(width, height, &settings)?;
    let progress = Progress::new();
    let progress_count = (rpp / 20).min(4).max(16);
    for ray in 0..(rpp - 1) {
        let _ = kernel.run(&settings, false)?;
        if ray > 0 && ray % progress_count == 0 {
            kernel.sync()?;
            let value = ray as f32 / rpp as f32;
            let mut seconds = progress.time(value);
            let minutes = (seconds / 60.0) as u32;
            seconds -= (minutes * 60) as f32;
            println!("{:.2}%, {}:{:.2} left", 100.0 * value, minutes, seconds);
        }
    }
    kernel.sync()?;
    println!("Last ray...");
    let image = kernel.run(&settings, true)?.unwrap();
    println!("render done, saving");
    save_image(&image, "render.png")?;
    println!("done");
    Ok(())
}

fn video_one(frame: u32, rpp: u32, kernel: &mut Kernel, settings: &Settings) -> Result<(), Error> {
    for _ in 0..(rpp - 1) {
        let _ = kernel.run(&settings, false)?;
    }
    let image = kernel.run(&settings, true)?.unwrap();
    save_image(&image, &format!("render{:03}.png", frame))?;
    Ok(())
}

pub fn video(width: u32, height: u32, rpp: u32, frames: u32) -> Result<(), Error> {
    let mut default_settings = Settings::new();
    default_settings.clear_constants();
    let mut kernel = Kernel::new(width, height, &default_settings)?;
    let mut keyframes = KeyframeList::new("keyframes.clam5", default_settings)?;
    let progress = Progress::new();
    for frame in 0..frames {
        let settings = keyframes.interpolate(frame as f32 / frames as f32);
        video_one(frame, rpp, &mut kernel, &settings)?;
        let value = (frame + 1) as f32 / frames as f32;
        let mut seconds = progress.time(value);
        let minutes = (seconds / 60.0) as u32;
        seconds -= (minutes * 60) as f32;
        println!("{:.2}%, {}:{:.2} left", 100.0 * value, minutes, seconds);
    }
    println!("done");
    Ok(())
}

fn try_render(args: &[String]) -> Result<(), Error> {
    if args.len() == 2 {
        let rpp = args[1].parse()?;
        match &*args[0] {
            "8k" => headless(3840 * 2, 2160 * 2, rpp),
            "4k" => headless(3840, 2160, rpp),
            "1080p" | "2k" => headless(1920, 1080, rpp),
            pix => headless(pix.parse()?, pix.parse()?, rpp),
        }
    } else {
        headless(args[0].parse()?, args[1].parse()?, args[2].parse()?)
    }
}

fn try_video(args: &[String]) -> Result<(), Error> {
    if args.len() == 4 {
        video(
            args[0].parse()?,
            args[1].parse()?,
            args[2].parse()?,
            args[3].parse()?,
        )
    } else {
        Err(failure::err_msg(
            "--video needs four args: [width] [height] [rpp] [frames]",
        ))
    }
}

fn render(args: &[String]) {
    match try_render(args) {
        Ok(()) => (),
        Err(err) => println!("{}", err),
    }
}

fn video_cmd(args: &[String]) {
    match try_video(args) {
        Ok(()) => (),
        Err(err) => println!("{}", err),
    }
}

fn interactive_cmd() {
    let width = 200;
    let height = 200;

    match display::display(width, height, &interactive) {
        Ok(()) => (),
        Err(err) => println!("{}", err),
    };
}

fn main() {
    let arguments = args().skip(1).collect::<Vec<_>>();
    if arguments.len() > 2 && arguments[0] == "--render" {
        render(&arguments[1..]);
    } else if arguments.len() > 2 && arguments[0] == "--video" {
        video_cmd(&arguments[1..]);
    } else {
        interactive_cmd();
    }
}
