use display::Image;
use display::ScreenEvent;
use failure::Error;
use failure;
use input::Input;
use mandelbox_cfg::MandelboxCfg;
use ocl;
use png;
use progress::Progress;
use settings::Settings;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::sync::{mpsc, Arc, Mutex};

const MANDELBOX: &str = include_str!("mandelbox.cl");
const DATA_WORDS: u32 = 5;

struct Kernel {
    context: ocl::Context,
    queue: ocl::Queue,
    kernel: ocl::Kernel,
    cpu_cfg: MandelboxCfg,
    data: Option<ocl::Buffer<u8>>,
    cfg: ocl::Buffer<MandelboxCfg>,
    width: u32,
    height: u32,
    frame: u32,
}

fn dump_binary(program: &ocl::Program) -> Result<(), Error> {
    if let Ok(path) = env::var("CLAM5_BINARY") {
        if let ocl::enums::ProgramInfoResult::Binaries(binaries) =
            program.info(ocl::enums::ProgramInfo::Binaries)?
        {
            if binaries.len() != 1 {
                for (i, binary) in binaries.iter().enumerate() {
                    let mut file = File::create(format!("{}.{}", &path, i))?;
                    file.write_all(&binary[..])?;
                }
            } else {
                let mut file = File::create(format!("{}", &path))?;
                file.write_all(&binaries[0][..])?;
            }
            println!("Dumped binaries");
        }
    }
    Ok(())
}

impl Kernel {
    fn new(width: u32, height: u32) -> Result<Kernel, Error> {
        let context = Self::make_context()?;
        let device = context.devices()[0];
        let device_name = device.name()?;
        let program = {
            let mut builder = ocl::Program::builder();
            builder.source(MANDELBOX);
            builder.devices(device);
            if device_name.contains("GeForce") {
                builder.cmplr_opt("-cl-nv-verbose");
            }
            builder.build(&context)?
        };
        println!("Using GPU: {}", device_name);
        if let ocl::enums::ProgramBuildInfoResult::BuildLog(log) =
            program.build_info(context.devices()[0], ocl::enums::ProgramBuildInfo::BuildLog)?
        {
            let log = log.trim();
            if !log.is_empty() {
                println!("{}", log);
            }
        }
        dump_binary(&program)?;
        let queue = ocl::Queue::new(&context, device, None)?;
        let cfg = ocl::Buffer::builder().context(&context).len(1).build()?;
        let kernel = ocl::Kernel::builder()
            .program(&program)
            .name("Main")
            .arg(None::<&ocl::Buffer<u8>>)
            .arg(&cfg)
            .arg(0u32)
            .arg(0u32)
            .arg(0u32)
            .arg(0u32)
            .build()?;
        Ok(Kernel {
            context,
            queue,
            kernel,
            data: None,
            cpu_cfg: MandelboxCfg::default(),
            cfg,
            width,
            height,
            frame: 0,
        })
    }

    fn make_context() -> Result<ocl::Context, Error> {
        let mut last_err = None;
        for platform in ocl::Platform::list() {
            match ocl::Context::builder()
                .platform(platform)
                .devices(ocl::DeviceType::new().gpu())
                .build()
            {
                Ok(ok) => return Ok(ok),
                Err(e) => last_err = Some(e),
            }
        }
        for platform in ocl::Platform::list() {
            match ocl::Context::builder().platform(platform).build() {
                Ok(ok) => return Ok(ok),
                Err(e) => last_err = Some(e),
            }
        }
        match last_err {
            Some(e) => Err(e.into()),
            None => Err(failure::err_msg("No OpenCL devices found")),
        }
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        self.width = width;
        self.height = height;
        self.data = None;
        self.frame = 0;
        Ok(())
    }

    fn set_args(&mut self, settings: &Settings, output_linear: bool) -> Result<(), Error> {
        let old_cfg = self.cpu_cfg;
        self.cpu_cfg.read(settings);
        if old_cfg != self.cpu_cfg {
            let to_write = [self.cpu_cfg];
            self.cfg.write(&to_write as &[_]).queue(&self.queue).enq()?;
            self.frame = 0;
        }
        let data = match self.data {
            Some(ref data) => data,
            None => {
                let data = ocl::Buffer::builder()
                    .context(&self.context)
                    .len(self.width * self.height * DATA_WORDS * 4)
                    .build()?;
                self.data = Some(data);
                self.data.as_ref().unwrap()
            }
        };
        self.kernel.set_arg(0, data)?;
        self.kernel.set_arg(1, &self.cfg)?;
        self.kernel.set_arg(2, self.width as u32)?;
        self.kernel.set_arg(3, self.height as u32)?;
        self.kernel.set_arg(4, self.frame as u32)?;
        self.kernel
            .set_arg(5, if output_linear { 1u32 } else { 0u32 })?;
        Ok(())
    }

    fn run(
        &mut self,
        settings: &Settings,
        download: bool,
        output_linear: bool,
    ) -> Result<Option<Image>, Error> {
        self.set_args(settings, output_linear)?;
        let lws = 1024;
        let to_launch = self.kernel
            .cmd()
            .queue(&self.queue)
            .global_work_size((self.width * self.height + lws - 1) / lws * lws);
        // enq() is unsafe, even though the Rust code is safe (unsafe due to untrusted GPU code)
        unsafe { to_launch.enq() }?;
        self.frame += 1;
        if download {
            let mut vec = vec![0u8; self.width as usize * self.height as usize * 4];
            self.data
                .as_ref()
                .unwrap()
                .read(&mut vec)
                .queue(&self.queue)
                .enq()?;
            let image = Image::new(vec, self.width, self.height);
            Ok(Some(image))
        } else {
            Ok(None)
        }
    }

    fn sync(&mut self) -> ocl::Result<()> {
        self.queue.finish()
    }
}

pub fn interactive(
    width: u32,
    height: u32,
    settings_input: &Arc<Mutex<(Settings, Input)>>,
    send_image: &mpsc::Sender<Image>,
    screen_events: &mpsc::Receiver<ScreenEvent>,
) -> Result<(), Error> {
    let mut kernel = Kernel::new(width, height)?;
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
            (*settings).clone()
        };
        let image = kernel.run(&settings, true, true)?.unwrap();
        match send_image.send(image) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        };
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
    let mut settings = ::settings::init_settings();
    ::settings::load_settings(&mut settings, "settings.clam5")?;
    let mut kernel = Kernel::new(width, height)?;
    let progress = Progress::new();
    let progress_count = (rpp / 20).min(4).max(16);
    for ray in 0..(rpp - 1) {
        let _ = kernel.run(&settings, false, false)?;
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
    let image = kernel.run(&settings, true, false)?.unwrap();
    println!("render done, saving");
    save_image(&image, "render.png")?;
    println!("done");
    Ok(())
}
