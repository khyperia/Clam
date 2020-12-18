mod fps_counter;
mod input;
mod interactive;
mod kernel;
mod keyframe_list;
mod progress;
mod render_window;
mod setting_value;
mod settings;
mod settings_input;
mod texture_blit;

use cgmath::Vector3;
// use chrono::prelude::*;
// use display::Key;
// use kernel::Kernel;
// use kernel_compilation::MANDELBOX;
// use keyframe_list::KeyframeList;
// use png::{BitDepth, ColorType, Encoder};
// use progress::Progress;
// use settings::Settings;
// use std::{
//     env::args,
//     fs::File,
//     io::{BufWriter, Write},
//     mem::drop,
//     process::{Command, Stdio},
//     str,
//     sync::mpsc,
// };

use winit::event::VirtualKeyCode as Key;

type Error = Box<dyn std::error::Error>;

fn parse_vector3(v: &str) -> Option<Vector3<f64>> {
    let mut split = v.split_ascii_whitespace();
    let x = split.next()?.parse().ok()?;
    let y = split.next()?.parse().ok()?;
    let z = split.next()?.parse().ok()?;
    if split.next().is_some() {
        None
    } else {
        Some(Vector3::new(x, y, z))
    }
}

fn cast_slice<A, B>(a: &[A]) -> &[B] {
    let new_len = std::mem::size_of_val(a) / std::mem::size_of::<B>();
    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const B, new_len) }
}

fn f32_to_u8(px: f32) -> u8 {
    (px * 255.0).max(0.0).min(255.0) as u8
}

/*
fn save_image(image: &CpuTexture<[f32; 4]>, path: &str) -> Result<(), Error> {
    let file = File::create(path)?;
    let w = &mut BufWriter::new(file);
    write_image(image, w)
}

fn write_image(image: &CpuTexture<[f32; 4]>, w: impl Write) -> Result<(), Error> {
    let mut encoder = Encoder::new(w, image.size.0 as u32, image.size.1 as u32);
    encoder.set_color(ColorType::RGB);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let (width, height) = image.size;
    let mut output = vec![0; width * height * 3];
    let data = image.data();
    for y in 0..height {
        for x in 0..width {
            let in_idx = y * width + x;
            let out_idx = (y * width + x) * 3;
            output[out_idx] = f32_to_u8(data[in_idx][0]);
            output[out_idx + 1] = f32_to_u8(data[in_idx][1]);
            output[out_idx + 2] = f32_to_u8(data[in_idx][2]);
        }
    }
    writer.write_image_data(&output)?;
    Ok(())
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
    let realized_source = MANDELBOX.get()?;
    let loaded_settings = Settings::load("settings.clam5", realized_source.default_settings())?;
    let mut kernel = Kernel::create(realized_source, width, height)?;
    let progress = Progress::new();
    let progress_count = progress_count(rpp);
    for ray in 0..rpp {
        kernel.run(&loaded_settings)?;
        if ray > 0 && ray % progress_count == 0 {
            kernel.sync_renderer()?;
            let value = ray as f64 / rpp as f64;
            println!("{}", progress.time_str(value));
        }
    }
    kernel.sync_renderer()?;
    println!("render done, downloading");
    let image = kernel.download()?;
    println!("saving, final time: {}", progress.time_str(1.0));
    let local: DateTime<Local> = Local::now();
    let filename = local.format("%Y-%m-%d_%H-%M-%S.png").to_string();
    save_image(&image, &filename)?;
    println!("done");
    Ok(())
}

enum VideoFormat {
    MP4,
    Twitter,
    PngSeq,
    Gif,
}

impl std::str::FromStr for VideoFormat {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("mp4") {
            Ok(VideoFormat::MP4)
        } else if s.eq_ignore_ascii_case("twitter") {
            Ok(VideoFormat::Twitter)
        } else if s.eq_ignore_ascii_case("pngseq") {
            Ok(VideoFormat::PngSeq)
        } else if s.eq_ignore_ascii_case("gif") {
            Ok(VideoFormat::Gif)
        } else {
            Err("Invalid video format".into())
        }
    }
}

fn video_one(
    rpp: usize,
    kernel: &mut Kernel<[f32; 4]>,
    settings: &Settings,
    stream: &mpsc::SyncSender<CpuTexture<[f32; 4]>>,
) -> Result<(), Error> {
    for _ in 0..rpp {
        kernel.run(settings)?;
    }
    let image = kernel.download()?;
    stream.send(image)?;
    Ok(())
}

fn ffmpeg(args: &[&str]) -> Result<(), Error> {
    let exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    if Command::new(exe).args(args).status()?.success() {
        Ok(())
    } else {
        Err("ffmpeg failed".into())
    }
}

fn do_gifize() -> Result<(), Error> {
    ffmpeg(&["-i", "%04d.png", "-vf", "palettegen", "-y", "palette.png"])?;
    ffmpeg(&[
        "-framerate",
        "50",
        "-i",
        "%04d.png",
        "-i",
        "palette.png",
        "-lavfi",
        "paletteuse",
        "-y",
        "output.gif",
    ])?;
    Ok(())
}

fn pngseq_write(stream: &mpsc::Receiver<CpuTexture<[f32; 4]>>, gifize: bool) -> Result<(), Error> {
    let mut i = 0;
    if gifize {
        for item in std::fs::read_dir(".")? {
            let item = item?;
            let is_num = item.path().file_stem().map_or(false, |x| {
                x.to_str().map_or(false, |x| x.parse::<u64>().is_ok())
            });
            if is_num {
                std::fs::remove_file(item.path())?;
            }
        }
    }
    while let Ok(img) = stream.recv() {
        save_image(&img, &format!("{:04}.png", i))?;
        i += 1;
    }
    if gifize {
        do_gifize()?;
    }
    Ok(())
}

fn video_write_from_pngseq(twitter: bool) -> Result<(), Error> {
    let mut args = Vec::new();
    args.extend_from_slice(&["-framerate", "60", "-i", "%04d.png"]);
    if twitter {
        args.extend_from_slice(&["-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "2048K"]);
    }
    args.extend_from_slice(&["video.mp4", "-y"]);
    ffmpeg(&args)
}

fn video_write(stream: &mpsc::Receiver<CpuTexture<[f32; 4]>>, twitter: bool) -> Result<(), Error> {
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
        Err(format!("ffmpeg exited with error code: {}", res).into())
    }
}

fn video(
    width: usize,
    height: usize,
    rpp: usize,
    frames: usize,
    wrap: bool,
    format: VideoFormat,
) -> Result<(), Error> {
    let realized_source = MANDELBOX.get()?;
    let keyframes = KeyframeList::load("keyframes.clam5", &realized_source)?;
    let mut kernel = Kernel::create(realized_source, width, height)?;
    let progress = Progress::new();

    let (send, recv) = mpsc::sync_channel(5);

    let thread_handle = match format {
        VideoFormat::PngSeq => {
            std::thread::spawn(move || pngseq_write(&recv, false).expect("Couldn't write frame"))
        }
        VideoFormat::Gif => {
            std::thread::spawn(move || pngseq_write(&recv, true).expect("Couldn't write frame"))
        }
        VideoFormat::MP4 => {
            std::thread::spawn(move || video_write(&recv, false).expect("Couldn't write frame"))
        }
        VideoFormat::Twitter => {
            std::thread::spawn(move || video_write(&recv, true).expect("Couldn't write frame"))
        }
    };

    for frame in 0..frames {
        let settings = keyframes.interpolate(frame as f64 / frames as f64, wrap);
        video_one(rpp, &mut kernel, &settings, &send)?;
        let value = (frame + 1) as f64 / frames as f64;
        println!("{}", progress.time_str(value));
    }
    drop(send);
    thread_handle.join().expect("Couldn't join thread");
    println!("done");
    Ok(())
}

fn parse_resolution(res: &str) -> Option<(usize, usize)> {
    if let Some(dash) = res.find('-') {
        let (x, y) = res.split_at(dash);
        let y = &y[1..];
        Some((x.parse().ok()?, y.parse().ok()?))
    } else {
        match res {
            "32k" => Some((30720, 17280)),
            "16k" => Some((15360, 8640)),
            "8k" => Some((7680, 4320)),
            "4k" => Some((3840, 2160)),
            "2k" => Some((1920, 1080)),
            "1k" => Some((960, 540)),
            "0.5k" => Some((480, 270)),
            "0.25k" => Some((240, 135)),
            "twitter" => Some((1280, 720)),
            _ => None,
        }
    }
}

fn render(args: &[String]) -> Result<(), Error> {
    if args.len() == 2 {
        let (width, height) = parse_resolution(&args[0]).ok_or_else(|| "Invalid resolution")?;
        let rpp = args[1].parse()?;
        image(width, height, rpp)
    } else {
        Err("--render needs two args: [width-height|0.25k..32k|twitter] [rpp]".into())
    }
}

fn video_cmd(args: &[String]) -> Result<(), Error> {
    if args.len() == 5 {
        let (width, height) = parse_resolution(&args[0]).ok_or_else(|| "Invalid resolution")?;
        let rpp = args[1].parse()?;
        let frames = args[2].parse()?;
        let wrap = args[3].parse()?;
        let format = args[4].parse()?;
        video(width, height, rpp, frames, wrap, format)
    } else {
        Err("--video needs four args: [width-height|0.25k..32k|twitter] [rpp] [frames] [wrap:true|false] [format:mp4|twitter|pngseq|gif]".into())
    }
}

fn pngseq(format: VideoFormat) -> Result<(), Error> {
    match format {
        VideoFormat::Gif => do_gifize(),
        VideoFormat::MP4 => video_write_from_pngseq(false),
        VideoFormat::Twitter => video_write_from_pngseq(false),
        VideoFormat::PngSeq => Err("--video needs one arg: [format:mp4|twitter|gif]".into()),
    }
}

fn pngseq_cmd(args: &[String]) -> Result<(), Error> {
    if args.len() == 1 {
        pngseq(args[0].parse()?)
    } else {
        Err("--video needs one arg: [format:mp4|twitter|gif]".into())
    }
}

fn main() -> Result<(), Error> {
    let arguments = args().skip(1).collect::<Vec<_>>();
    if arguments.len() > 2 && &arguments[0] == "--render" {
        display::run_headless(|| render(&arguments[1..]))??
    } else if arguments.len() > 2 && &arguments[0] == "--video" {
        display::run_headless(|| video_cmd(&arguments[1..]))??
    } else if arguments.len() == 2 && &arguments[0] == "--pngseq" {
        pngseq_cmd(&arguments[1..])?
    } else if arguments.is_empty() {
        display_gl::run(1920.0, 1080.0)?
    } else {
        println!("Usage:");
        println!("clam5 --render [width-height|0.25k..32k|twitter] [rpp]");
        println!("clam5 --video [width-height|0.25k..32k|twitter] [rpp] [frames] [wrap:true|false] [format:mp4|twitter|pngseq|gif]");
        println!("clam5 --pngseq [format:mp4|twitter|gif]");
        println!("clam5");
    }
    Ok(())
}
*/

fn main() {
    env_logger::init();
    render_window::run();
}
