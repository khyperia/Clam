mod display;
mod display_gl;
#[cfg(feature = "vr")]
mod display_vr;
mod fps_counter;
mod gl_help;
mod input;
mod interactive;
mod kernel;
mod kernel_compilation;
mod progress;
mod render_text;
mod render_texture;
mod setting_value;
mod settings;

use chrono::prelude::*;
use failure::{err_msg, Error};
use gl::types::*;
use gl_help::CpuTexture;
use glutin::event::VirtualKeyCode as Key;
use kernel::FractalKernel;
use png::{BitDepth, ColorType, Encoder};
use progress::Progress;
use settings::{KeyframeList, Settings};
use std::{
    env::args,
    ffi::c_void,
    fs::File,
    io::{BufWriter, Write},
    mem::drop,
    process::{Command, Stdio},
    ptr::null,
    slice, str,
    sync::mpsc,
};

fn f32_to_u8(px: f32) -> u8 {
    (px * 255.0).max(0.0).min(255.0) as u8
}

fn save_image(image: &CpuTexture<[f32; 4]>, path: &str) -> Result<(), Error> {
    let file = File::create(path)?;
    let w = &mut BufWriter::new(file);
    write_image(image, w)
}

fn write_image(image: &CpuTexture<[f32; 4]>, w: impl Write) -> Result<(), Error> {
    let mut encoder = Encoder::new(w, image.width as u32, image.height as u32);
    encoder.set_color(ColorType::RGB);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let width = image.width;
    let height = image.height;
    let mut output = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let in_idx = y * width + x;
            let out_idx = ((height - y - 1) * width + x) * 3;
            output[out_idx] = f32_to_u8(image.data[in_idx][0]);
            output[out_idx + 1] = f32_to_u8(image.data[in_idx][1]);
            output[out_idx + 2] = f32_to_u8(image.data[in_idx][2]);
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
    source: GLenum,
    type_: GLenum,
    id: GLuint,
    severity: GLenum,
    length: GLsizei,
    message: *const GLchar,
    _: *mut c_void,
) {
    let msg =
        str::from_utf8(unsafe { slice::from_raw_parts(message as *const u8, length as usize) });
    println!(
        "GL debug callback: source:{} type:{} id:{} severity:{} {:?}",
        source, type_, id, severity, msg
    );
}

#[cfg(not(windows))]
fn progress_count(rpp: usize) -> usize {
    (rpp / 20).min(4).max(16)
}

#[cfg(windows)]
fn progress_count(_: usize) -> usize {
    1
}

fn image(width: usize, height: usize, rpp: usize) -> Result<(), Error> {
    let mut settings = Settings::new();
    let mut kernel = FractalKernel::create(width, height, &mut settings)?;
    settings.load("settings.clam5")?;
    settings.all_constants();
    kernel.rebuild(&mut settings, false)?;
    let progress = Progress::new();
    let progress_count = progress_count(rpp);
    for ray in 0..rpp {
        kernel.run(&mut settings, false)?;
        if ray > 0 && ray % progress_count == 0 {
            kernel.sync_renderer()?;
            let value = ray as f64 / rpp as f64;
            println!("{}", progress.time_str(value));
        }
    }
    kernel.sync_renderer()?;
    let image = kernel.download()?;
    println!("render done, saving");
    let local: DateTime<Local> = Local::now();
    let filename = local.format("%Y-%m-%d_%H-%M-%S.png").to_string();
    save_image(&image, &filename)?;
    println!("done");
    Ok(())
}

fn video_one(
    rpp: usize,
    kernel: &mut FractalKernel<[f32; 4]>,
    settings: &mut Settings,
    stream: &mpsc::SyncSender<CpuTexture<[f32; 4]>>,
) -> Result<(), Error> {
    for _ in 0..rpp {
        kernel.run(settings, false)?;
    }
    let image = kernel.download()?;
    stream.send(image)?;
    Ok(())
}

fn video_write(stream: mpsc::Receiver<CpuTexture<[f32; 4]>>, twitter: bool) -> Result<(), Error> {
    let exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let mut ffmpeg = Command::new(exe);
    ffmpeg.stdin(Stdio::piped());
    ffmpeg.args(&["-f", "image2pipe", "-framerate", "60", "-i", "-"]);
    if twitter {
        ffmpeg.args(&["-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "2048K"]);
    }
    ffmpeg.args(&["video.mp4", "-y"]);
    let mut ffmpeg = ffmpeg.spawn()?;
    while let Ok(img) = stream.recv() {
        let ffmpeg_stdin = ffmpeg
            .stdin
            .as_mut()
            .expect("ffmpeg process failed to redirect stdin");
        write_image(&img, ffmpeg_stdin)?;
    }
    // make sure to drop stdin to close process before waiting
    ffmpeg.stdin = None;
    let res = ffmpeg.wait()?;
    if res.success() {
        Ok(())
    } else {
        Err(err_msg(format!("ffmpeg exited with error code: {}", res)))
    }
}

fn video(
    width: usize,
    height: usize,
    rpp: usize,
    frames: usize,
    wrap: bool,
    twitter: bool,
) -> Result<(), Error> {
    let mut default_settings = Settings::new();
    let mut kernel = FractalKernel::create(width, height, &mut default_settings)?;
    default_settings.clear_constants();
    let mut keyframes = KeyframeList::new("keyframes.clam5", default_settings)?;
    let progress = Progress::new();

    let (send, recv) = mpsc::sync_channel(5);

    let thread_handle = std::thread::spawn(move || video_write(recv, twitter));

    for frame in 0..frames {
        let settings = keyframes.interpolate(frame as f32 / frames as f32, wrap);
        video_one(rpp, &mut kernel, settings, &send)?;
        let value = (frame + 1) as f64 / frames as f64;
        println!("{}", progress.time_str(value));
    }
    drop(send);
    thread_handle.join().expect("Couldn't join thread")?;
    println!("done");
    Ok(())
}

fn render(args: &[String]) -> Result<(), Error> {
    if args.len() == 2 {
        let rpp = args[1].parse()?;
        match &*args[0] {
            "32k" => image(30720, 17280, rpp),
            "16k" => image(15360, 8640, rpp),
            "8k" => image(7680, 4320, rpp),
            "4k" => image(3840, 2160, rpp),
            "2k" => image(1920, 1080, rpp),
            "1k" => image(960, 540, rpp),
            "0.5k" => image(480, 270, rpp),
            "0.25k" => image(240, 135, rpp),
            pix => image(pix.parse()?, pix.parse()?, rpp),
        }
    } else if args.len() == 3 {
        image(args[0].parse()?, args[1].parse()?, args[2].parse()?)
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
            false,
        )
    } else if args.len() == 4 {
        let rpp = args[1].parse()?;
        let frames = args[2].parse()?;
        let wrap = args[3].parse()?;
        match &*args[0] {
            "32k" => video(30720, 17280, rpp, frames, wrap, false),
            "16k" => video(15360, 8640, rpp, frames, wrap, false),
            "8k" => video(7680, 4320, rpp, frames, wrap, false),
            "4k" => video(3840, 2160, rpp, frames, wrap, false),
            "2k" => video(1920, 1080, rpp, frames, wrap, false),
            "1k" => video(960, 540, rpp, frames, wrap, false),
            "0.5k" => video(480, 270, rpp, frames, wrap, false),
            "0.25k" => video(240, 135, rpp, frames, wrap, false),
            "twitter" => video(1280, 720, rpp, frames, wrap, true),
            _ => Err(failure::err_msg("Invalid video resolution alias")),
        }
    } else {
        Err(failure::err_msg(
            "--video needs four or five args: [[width] [height]|[0.25k..32k|twitter]] [rpp] [frames] [wrap:true|false]",
        ))
    }
}

fn try_main() -> Result<(), Error> {
    let arguments = args().skip(1).collect::<Vec<_>>();
    if arguments.len() > 2 && arguments[0] == "--render" {
        display::run_headless(|| render(&arguments[1..]))??
    } else if arguments.len() > 2 && arguments[0] == "--video" {
        display::run_headless(|| video_cmd(&arguments[1..]))??
    } else if cfg!(feature = "vr") && arguments.len() == 1 && arguments[0] == "--vr" {
        #[cfg(feature = "vr")]
        display_vr::vr_display()?
    } else if arguments.is_empty() {
        display_gl::gl_display(1920.0, 1080.0)?
    } else {
        println!("Usage:");
        println!("clam5 --render [width] [height] [rpp]");
        println!("clam5 --render [0.25k..32k] [rpp]");
        println!("clam5 --video [width] [height] [rpp] [frames] [wrap:true|false]");
        println!("clam5 --video [0.25k..32k] [rpp] [frames] [wrap:true|false]");
        println!("clam5 --video twitter [rpp] [frames] [wrap:true|false]");
        #[cfg(feature = "vr")]
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
