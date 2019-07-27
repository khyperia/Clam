#[macro_use]
extern crate lazy_static;
extern crate byteorder;
extern crate cgmath;
extern crate failure;
extern crate gl;
extern crate ocl;
#[cfg(windows)]
extern crate openvr;
extern crate png;
extern crate regex;
extern crate sdl2;

mod display_gl;
#[cfg(windows)]
mod display_vr;
mod fps_counter;
mod input;
mod interactive;
mod kernel;
mod kernel_compilation;
mod progress;
mod render_text;
mod render_texture;
mod setting_value;
mod settings;

use failure::Error;
use interactive::ImageData;
use kernel::FractalKernel;
use png::BitDepth;
use png::ColorType;
use png::Encoder;
use progress::Progress;
use settings::KeyframeList;
use settings::Settings;
use std::env::args;
use std::ffi::c_void;
use std::fs::File;
use std::io::BufWriter;
use std::ptr::null;
use std::slice;
use std::str;
use std::sync::mpsc;

fn f32_to_u8(px: f32) -> u8 {
    (px * 255.0).max(0.0).min(255.0) as u8
}

fn save_image(image: &ImageData<f32>, path: &str) -> Result<(), Error> {
    let file = File::create(path)?;
    let w = &mut BufWriter::new(file);
    let mut encoder = Encoder::new(w, image.width, image.height);
    encoder.set_color(ColorType::RGB);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let width = image.width as usize;
    let height = image.height as usize;
    let input = image
        .data_cpu
        .as_ref()
        .expect("save_image must have cpu image");
    let mut output = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let in_idx = (y * width + x) * 4;
            let out_idx = ((height - y - 1) * width + x) * 3;
            output[out_idx] = f32_to_u8(input[in_idx]);
            output[out_idx + 1] = f32_to_u8(input[in_idx + 1]);
            output[out_idx + 2] = f32_to_u8(input[in_idx + 2]);
        }
    }
    writer.write_image_data(&output)?;
    Ok(())
}

fn check_gl() -> Result<(), Error> {
    // This should technically loop.
    let er = unsafe { gl::GetError() };
    if er == gl::NO_ERROR {
        return Ok(());
    }
    Err(failure::err_msg(format!("OGL error: {}", er)))
}

fn gl_register_debug() -> Result<(), Error> {
    unsafe {
        gl::DebugMessageCallback(debug_callback, null());
    }
    check_gl()?;
    Ok(())
}

extern "system" fn debug_callback(
    source: u32,
    type_: u32,
    id: u32,
    severity: u32,
    length: i32,
    message: *const i8,
    _: *mut c_void,
) {
    let msg =
        str::from_utf8(unsafe { slice::from_raw_parts(message as *const u8, length as usize) });
    println!(
        "GL debug callback: source:{} type:{} id:{} severity:{} {:?}",
        source, type_, id, severity, msg
    );
}

#[cfg(windows)]
#[link(name = "Shell32")]
extern "C" {}

#[cfg(not(windows))]
fn progress_count(rpp: u32) -> u32 {
    (rpp / 20).min(4).max(16)
}

#[cfg(windows)]
fn progress_count(_: u32) -> u32 {
    1
}

fn headless(width: u32, height: u32, rpp: u32) -> Result<(), Error> {
    let mut settings = Settings::new();
    let mut kernel = FractalKernel::create(width, height, false, &mut settings)?;
    settings.load("settings.clam5")?;
    settings.all_constants();
    let progress = Progress::new();
    let progress_count = progress_count(rpp);
    for ray in 0..rpp {
        kernel.run(&mut settings, false)?;
        if ray > 0 && ray % progress_count == 0 {
            kernel.sync_renderer()?;
            let value = ray as f32 / rpp as f32;
            println!("{}", progress.time_str(value));
        }
    }
    kernel.sync_renderer()?;
    let image = kernel.download()?;
    println!("render done, saving");
    save_image(&image, "render.png")?;
    println!("done");
    Ok(())
}

fn video_one(
    frame: u32,
    rpp: u32,
    kernel: &mut FractalKernel<f32>,
    settings: &mut Settings,
    stream: &mpsc::SyncSender<(u32, ImageData<f32>)>,
) -> Result<(), Error> {
    for _ in 0..rpp {
        kernel.run(settings, false)?;
    }
    let image = kernel.download()?;
    stream.send((frame, image))?;
    Ok(())
}

fn video(width: u32, height: u32, rpp: u32, frames: u32, wrap: bool) -> Result<(), Error> {
    let mut default_settings = Settings::new();
    let mut kernel = FractalKernel::create(width, height, false, &mut default_settings)?;
    default_settings.clear_constants();
    let mut keyframes = KeyframeList::new("keyframes.clam5", default_settings)?;
    let progress = Progress::new();

    let (send, recv) = mpsc::sync_channel(5);

    std::thread::spawn(move || {
        while let Ok((frame, img)) = recv.recv() {
            match save_image(&img, &format!("render{:03}.png", frame)) {
                Ok(()) => (),
                Err(err) => println!("Error saving image: {}", err),
            }
        }
    });

    for frame in 0..frames {
        let settings = keyframes.interpolate(frame as f32 / frames as f32, wrap);
        video_one(frame, rpp, &mut kernel, settings, &send)?;
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
            "32k" => headless(30720, 17280, rpp),
            "16k" => headless(15360, 8640, rpp),
            "8k" => headless(7680, 4320, rpp),
            "4k" => headless(3840, 2160, rpp),
            "2k" => headless(1920, 1080, rpp),
            "1k" => headless(960, 540, rpp),
            "0.5k" => headless(480, 270, rpp),
            "0.25k" => headless(240, 135, rpp),
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
    if args.len() == 5 {
        video(
            args[0].parse()?,
            args[1].parse()?,
            args[2].parse()?,
            args[3].parse()?,
            args[4].parse()?,
        )
    } else {
        Err(failure::err_msg(
            "--video needs five args: [width] [height] [rpp] [frames] [wrap:true|false]",
        ))
    }
}

fn interactive_cmd() -> Result<(), Error> {
    let width = 1920;
    let height = 1080;
    display_gl::gl_display(width, height)
}

fn try_main() -> Result<(), Error> {
    let arguments = args().skip(1).collect::<Vec<_>>();
    if arguments.len() > 2 && arguments[0] == "--render" {
        render(&arguments[1..])?;
    } else if arguments.len() > 2 && arguments[0] == "--video" {
        video_cmd(&arguments[1..])?;
    } else if cfg!(windows) && arguments.len() == 1 && arguments[0] == "--vr" {
        #[cfg(windows)]
        display_vr::vr_display()?;
    } else if arguments.is_empty() {
        interactive_cmd()?;
    } else {
        println!("Usage:");
        println!("clam5 --render [width] [height] [rpp]");
        println!("clam5 --render [8k|4k|2k|1k] [rpp]");
        println!("clam5 --video [width] [height] [rpp] [frames] [wrap:true|false]");
        println!("clam5 --vr");
        println!("clam5");
    }
    Ok(())
}

fn main() {
    match try_main() {
        Ok(()) => (),
        Err(err) => println!("Error in main: {}\n{}", err, err.backtrace()),
    }
}
