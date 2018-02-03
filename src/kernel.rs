use display;
use glium;
use input::Input;
use mandelbox_cfg::MandelboxCfg;
use ocl;
use ocl_core::types::abs::AsMem;
use png;
use progress::Progress;
use settings::Settings;
use std::error::Error;
use std::sync::{mpsc, Arc, Mutex};

const MANDELBOX: &str = include_str!("mandelbox.cl");
const DATA_WORDS: u32 = 3;

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

impl Kernel {
    fn new(width: u32, height: u32) -> Result<Kernel, Box<Error>> {
        let context = Self::make_context()?;
        let program = ocl::Program::builder().src(MANDELBOX).build(&context)?;
        if let ocl::enums::ProgramBuildInfoResult::BuildLog(log) =
            program.build_info(context.devices()[0], ocl::enums::ProgramBuildInfo::BuildLog)
        {
            let log = log.trim();
            if !log.is_empty() {
                println!("{}", log);
            }
        }
        let queue = ocl::Queue::new(&context, context.devices()[0], None)?;
        println!("{}", queue.device().name());
        let kernel = ocl::Kernel::new("Main", &program)?;
        let cfg = ocl::Buffer::builder().context(&context).dims(1).build()?;
        Ok(Kernel {
            context: context,
            queue: queue,
            kernel: kernel,
            data: None,
            cpu_cfg: MandelboxCfg::default(),
            cfg: cfg,
            width: width,
            height: height,
            frame: 0,
        })
    }

    fn make_context() -> Result<ocl::Context, Box<Error>> {
        let mut last_err = None;
        for platform in ocl::Platform::list() {
            match ocl::Context::builder()
                .platform(platform)
                .devices(ocl::DeviceType::new().gpu())
                .build() {
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
            None => Err("No OpenCL devices found".into()),
        }
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), Box<Error>> {
        self.width = width;
        self.height = height;
        self.data = None;
        self.frame = 0;
        Ok(())
    }

    fn set_args(&mut self, settings: &Settings) -> Result<(), Box<Error>> {
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
                    .dims(self.width * self.height * DATA_WORDS * 4)
                    .build()?;
                self.data = Some(data);
                self.data.as_ref().unwrap()
            }
        };
        unsafe {
            self.kernel.set_arg_unchecked::<usize>(
                0,
                ocl::enums::KernelArg::Mem(data.as_mem()),
            )?;
            self.kernel.set_arg_unchecked::<usize>(
                1,
                ocl::enums::KernelArg::Mem(self.cfg.as_mem()),
            )?;
            self.kernel.set_arg_unchecked(
                2,
                ocl::enums::KernelArg::Scalar(self.width),
            )?;
            self.kernel.set_arg_unchecked(
                3,
                ocl::enums::KernelArg::Scalar(self.height),
            )?;
            self.kernel.set_arg_unchecked(
                4,
                ocl::enums::KernelArg::Scalar(self.frame),
            )?;
        };
        Ok(())
    }

    fn run(
        &mut self,
        settings: &Settings,
        download: bool,
    ) -> Result<Option<glium::texture::RawImage2d<'static, u8>>, Box<Error>> {
        self.set_args(settings)?;
        let lws = 1024;
        let to_launch = self.kernel
            .cmd()
            .queue(&self.queue)
            .gws((self.width * self.height + lws - 1) / lws * lws);
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
            let image = glium::texture::RawImage2d::from_raw_rgba(vec, (self.width, self.height));
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
    settings_input: Arc<Mutex<(Settings, Input)>>,
    send_image: &mpsc::Sender<glium::texture::RawImage2d<'static, u8>>,
    screen_events: &mpsc::Receiver<display::ScreenEvent>,
    events_loop: Option<glium::glutin::EventsLoopProxy>,
) -> Result<(), Box<Error>> {
    let mut kernel = Kernel::new(width, height)?;
    loop {
        loop {
            let event = match screen_events.try_recv() {
                Ok(event) => event,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            };

            match event {
                display::ScreenEvent::Resize(width, height) => kernel.resize(width, height)?,
            }
        }

        let settings = {
            let mut locked = settings_input.lock().unwrap();
            let (ref mut settings, ref mut input) = *locked;
            input.integrate(settings);
            (*settings).clone()
        };
        let image = kernel.run(&settings, true)?.unwrap();
        match send_image.send(image) {
            Ok(()) => (),
            Err(_) => return Ok(()),
        };
        if let Some(ref events_loop) = events_loop {
            events_loop.wakeup()?
        }
    }
}

fn save_image(image: &glium::texture::RawImage2d<u8>, path: &str) -> Result<(), Box<Error>> {
    use png::HasParameters;
    let file = ::std::fs::File::create(path)?;
    let w = &mut ::std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, image.width, image.height);
    encoder.set(png::ColorType::RGBA).set(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&image.data)?;
    Ok(())
}

pub fn headless(width: u32, height: u32, rpp: u32) -> Result<(), Box<Error>> {
    let mut settings = ::settings::init_settings();
    ::settings::load_settings(&mut settings, "settings.clam5")?;
    let mut kernel = Kernel::new(width, height)?;
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
