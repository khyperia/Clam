extern crate failure;
extern crate ocl;
extern crate png;
extern crate sdl2;

mod display;
mod fps_counter;
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
use settings::KeyframeList;
use settings::Settings;
use std::env::args;
use std::sync::{mpsc, Arc, Mutex};

pub fn interactive(
    width: u32,
    height: u32,
    settings_input: &Arc<Mutex<(Settings, Input)>>,
    send_image: &mpsc::SyncSender<Image>,
    screen_events: &mpsc::Receiver<ScreenEvent>,
) -> Result<(), Error> {
    let mut kernel = Kernel::new(width, height, &settings_input.lock().unwrap().0)?;
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
            Err(mpsc::SendError(_)) => return Ok(()),
        };
    }
}

fn save_image(image: &Image, path: &str) -> Result<(), Error> {
    use png::HasParameters;
    let file = ::std::fs::File::create(path)?;
    let w = &mut ::std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, image.width, image.height);
    encoder.set(png::ColorType::RGB).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let sans_alpha = image
        .data
        .iter()
        .enumerate()
        .filter_map(|(index, &element)| if index % 4 == 3 { None } else { Some(element) })
        .collect::<Vec<_>>();
    writer.write_image_data(&sans_alpha)?;
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
            println!("{}", progress.time_str(value));
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
        println!("{}", progress.time_str(value));
    }
    println!("done");
    Ok(())
}

fn render(args: &[String]) -> Result<(), Error> {
    if args.len() == 2 {
        let rpp = args[1].parse()?;
        match &*args[0] {
            "16k" => headless(15360, 8640, rpp),
            "8k" => headless(7680, 4320, rpp),
            "4k" => headless(3840, 2160, rpp),
            "2k" => headless(1920, 1080, rpp),
            "1k" => headless(960, 540, rpp),
            pix => headless(pix.parse()?, pix.parse()?, rpp),
        }
    } else if args.len() == 3 {
        headless(args[0].parse()?, args[1].parse()?, args[2].parse()?)
    } else {
        Err(failure::err_msg(
            "--render needs two or three args: [width] [height] [rpp], or, [16k|8k|4k|2k|1k] [rpp]",
        ))
    }
}

fn video_cmd(args: &[String]) -> Result<(), Error> {
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

fn interactive_cmd() -> Result<(), Error> {
    let width = 200;
    let height = 200;
    display::display(width, height, &interactive)
}

fn main() -> Result<(), Error> {
    let arguments = args().skip(1).collect::<Vec<_>>();
    if arguments.len() > 2 && arguments[0] == "--render" {
        render(&arguments[1..])?;
    } else if arguments.len() > 2 && arguments[0] == "--video" {
        video_cmd(&arguments[1..])?;
    } else if arguments.is_empty() {
        interactive_cmd()?;
    } else {
        println!("Usage:");
        println!("clam5 --render [width] [height] [rpp]");
        println!("clam5 --render [8k|4k|2k|1k] [rpp]");
        println!("clam5 --video [width] [height] [rpp] [frames]");
        println!("clam5");
    }
    Ok(())
}
