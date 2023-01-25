mod buffer_blit;
mod fps_counter;
mod input;
mod interactive;
mod kernel;
mod kernel_uniforms;
mod keyframe_list;
mod progress;
mod render_window;
mod setting_value;
mod settings;
mod settings_input;

use cgmath::Vector3;
use chrono::prelude::*;
use kernel::Kernel;
use keyframe_list::KeyframeList;
use log::info;
use png::{BitDepth, ColorType, Encoder};
use progress::Progress;
use settings::Settings;
use std::{
    env::args,
    fs::File,
    io::{BufWriter, Write},
    mem::drop,
    process::{Command, Stdio},
    str,
    sync::mpsc,
};

use winit::event::VirtualKeyCode as Key;

pub type Error = Box<dyn std::error::Error>;

pub struct CpuTexture {
    data: Vec<u8>,
    size: (u32, u32),
}

impl CpuTexture {
    fn rgba_to_rgb(&mut self) {
        let mut index = 0;
        self.data.retain(|_| {
            index += 1;
            if index == 4 {
                index = 0;
                false
            } else {
                true
            }
        })
    }
}

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

fn save_image(image: &CpuTexture, path: &str) -> Result<(), Error> {
    let file = File::create(path)?;
    let w = &mut BufWriter::new(file);
    write_image(image, w)
}

fn write_image(image: &CpuTexture, w: impl Write) -> Result<(), Error> {
    let mut encoder = Encoder::new(w, image.size.0, image.size.1);
    encoder.set_color(ColorType::RGB);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&image.data)?;
    Ok(())
}

#[cfg(not(windows))]
fn progress_count(rpp: usize) -> usize {
    (rpp / 20).min(4).max(16)
}

// Special windows handling for TDR
#[cfg(windows)]
fn progress_count(_: usize) -> usize {
    1
}

fn image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rpp: usize,
) -> Result<(), Error> {
    let loaded_settings = Settings::load("settings.clam5", &Settings::get_default())?;
    let mut kernel = Kernel::create(device, queue, width, height);
    let progress = Progress::new();
    let progress_count = progress_count(rpp);
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for ray in 0..rpp {
        if ray > 0 && ray % progress_count == 0 {
            queue.submit(std::iter::once(encoder.finish()));
            encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            device.poll(wgpu::Maintain::Wait);
            let value = ray as f64 / rpp as f64;
            info!("{}", progress.time_str(value));
        }
        kernel.run(device, &mut encoder, &loaded_settings);
    }
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);
    info!("render done, downloading");
    let image = kernel.download(device, queue);
    info!("saving, final time: {}", progress.time_str(1.0));
    let local: DateTime<Local> = Local::now();
    let filename = local.format("%Y-%m-%d_%H-%M-%S.png").to_string();
    save_image(&image, &filename)?;
    info!("done");
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
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rpp: usize,
    kernel: &mut Kernel,
    settings: &Settings,
    stream: &mpsc::SyncSender<CpuTexture>,
) -> Result<(), Error> {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    for i in 0..rpp {
        if cfg!(windows) && i % 64 == 0 {
            queue.submit(std::iter::once(encoder.finish()));
            encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            device.poll(wgpu::Maintain::Wait);
        }
        kernel.run(device, &mut encoder, settings);
    }
    queue.submit(std::iter::once(encoder.finish()));
    let image = kernel.download(device, queue);
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

fn pngseq_write(stream: &mpsc::Receiver<CpuTexture>, gifize: bool) -> Result<(), Error> {
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
    } else {
        // video is corrupted otherwise for some reason
        args.extend_from_slice(&["-pix_fmt", "yuv420p"]);
    }
    args.extend_from_slice(&["video.mp4", "-y"]);
    ffmpeg(&args)
}

fn video_write(stream: &mpsc::Receiver<CpuTexture>, twitter: bool) -> Result<(), Error> {
    let exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let mut ffmpeg = Command::new(exe);
    ffmpeg.stdin(Stdio::piped());
    ffmpeg.args(["-f", "image2pipe", "-framerate", "60", "-i", "-"]);
    if twitter {
        ffmpeg.args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "2048K"]);
    }
    ffmpeg.args(["video.mp4", "-y"]);
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

#[allow(clippy::too_many_arguments)]
fn video(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rpp: usize,
    frames: usize,
    wrap: bool,
    format: VideoFormat,
) -> Result<(), Error> {
    let keyframes = KeyframeList::load("keyframes.clam5", Settings::get_default())?;
    let mut kernel = Kernel::create(device, queue, width, height);
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
        video_one(device, queue, rpp, &mut kernel, &settings, &send)?;
        let value = (frame + 1) as f64 / frames as f64;
        info!("{}", progress.time_str(value));
    }
    drop(send);
    thread_handle.join().expect("Couldn't join thread");
    info!("done");
    Ok(())
}

fn parse_resolution(res: &str) -> Option<(u32, u32)> {
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

async fn render(args: &[String]) -> Result<(), Error> {
    if args.len() == 2 {
        let (width, height) = parse_resolution(&args[0]).ok_or("Invalid resolution")?;
        let rpp = args[1].parse()?;
        let (device, queue) = render_window::run_headless().await;
        image(&device, &queue, width, height, rpp)
    } else {
        Err("--render needs two args: [width-height|0.25k..32k|twitter] [rpp]".into())
    }
}

async fn video_cmd(args: &[String]) -> Result<(), Error> {
    if args.len() == 5 {
        let (width, height) = parse_resolution(&args[0]).ok_or("Invalid resolution")?;
        let rpp = args[1].parse()?;
        let frames = args[2].parse()?;
        let wrap = args[3].parse()?;
        let format = args[4].parse()?;
        let (device, queue) = render_window::run_headless().await;
        video(&device, &queue, width, height, rpp, frames, wrap, format)
    } else {
        Err("--video needs four args: [width-height|0.25k..32k|twitter] [rpp] [frames] [wrap:true|false] [format:mp4|twitter|pngseq|gif]".into())
    }
}

fn pngseq(format: VideoFormat) -> Result<(), Error> {
    match format {
        VideoFormat::Gif => do_gifize(),
        VideoFormat::MP4 => video_write_from_pngseq(false),
        VideoFormat::Twitter => video_write_from_pngseq(true),
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

pub async fn run() -> Result<(), Error> {
    let arguments = args().skip(1).collect::<Vec<_>>();
    if arguments.len() > 2 && &arguments[0] == "--render" {
        render(&arguments[1..]).await?
    } else if arguments.len() > 2 && &arguments[0] == "--video" {
        video_cmd(&arguments[1..]).await?
    } else if arguments.len() == 2 && &arguments[0] == "--pngseq" {
        pngseq_cmd(&arguments[1..])?
    } else if arguments.is_empty() {
        if let Ok(window) = render_window::RenderWindow::new().await {
            window.run()
        }
    } else {
        info!("Usage:");
        info!("clam5 --render [width-height|0.25k..32k|twitter] [rpp]");
        info!("clam5 --video [width-height|0.25k..32k|twitter] [rpp] [frames] [wrap:true|false] [format:mp4|twitter|pngseq|gif]");
        info!("clam5 --pngseq [format:mp4|twitter|gif]");
        info!("clam5");
    }
    Ok(())
}
