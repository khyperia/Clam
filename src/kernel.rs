use display::Image;
use failure;
use failure::Error;
use mandelbox_cfg::MandelboxCfg;
use ocl;
use settings::SettingValue;
use settings::Settings;
use std::env;
use std::fs::File;
use std::io::prelude::*;

const MANDELBOX: &str = include_str!("mandelbox.cl");
const DATA_WORDS: u32 = 4;

pub struct Kernel {
    context: ocl::Context,
    queue: ocl::Queue,
    kernel: ocl::Kernel,
    data: Option<ocl::Buffer<u8>>,
    cpu_cfg: MandelboxCfg,
    cfg: ocl::Buffer<MandelboxCfg>,
    old_settings: Settings,
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
                let mut file = File::create(path.to_string())?;
                file.write_all(&binaries[0][..])?;
            }
            println!("Dumped binaries");
        }
    }
    Ok(())
}

impl Kernel {
    pub fn new(width: u32, height: u32, settings: &Settings) -> Result<Kernel, Error> {
        let context = Self::make_context()?;
        let device = context.devices()[0];
        let device_name = device.name()?;
        println!("Using device: {}", device_name);
        let queue = ocl::Queue::new(&context, device, None)?;
        let kernel = Self::rebuild(&queue, settings)?;
        let cfg = ocl::Buffer::builder().context(&context).len(1).build()?;
        Ok(Kernel {
            context,
            queue,
            kernel,
            data: None,
            cpu_cfg: MandelboxCfg::default(),
            cfg,
            old_settings: settings.clone(),
            width,
            height,
            frame: 0,
        })
    }

    pub fn rebuild_self(&mut self, settings: &Settings) -> Result<(), Error> {
        self.kernel = Kernel::rebuild(&self.queue, settings)?;
        Ok(())
    }

    fn rebuild(queue: &ocl::Queue, settings: &Settings) -> Result<ocl::Kernel, Error> {
        let program = {
            let mut builder = ocl::Program::builder();
            builder.source(MANDELBOX);
            builder.devices(queue.device());
            builder.cmplr_opt("-cl-fast-relaxed-math");
            let device_name = queue.device().name()?;
            if device_name.contains("GeForce") {
                builder.cmplr_opt("-cl-nv-verbose");
            }
            for key in settings.constants() {
                let value = settings.get(&key).unwrap();
                match *value {
                    SettingValue::F32(value, _) => {
                        builder.cmplr_opt(format!("-D {}={:.16}f", key, value))
                    }
                    SettingValue::U32(value) => builder.cmplr_opt(format!("-D {}={}", key, value)),
                };
            }
            builder.build(&queue.context())?
        };
        if let ocl::enums::ProgramBuildInfoResult::BuildLog(log) =
            program.build_info(queue.device(), ocl::enums::ProgramBuildInfo::BuildLog)?
        {
            let log = log.trim();
            if !log.is_empty() {
                println!("{}", log);
            }
        }
        dump_binary(&program)?;
        let kernel = ocl::Kernel::builder()
            .program(&program)
            .name("Main")
            .arg(None::<&ocl::Buffer<u8>>)
            .arg(None::<&ocl::Buffer<MandelboxCfg>>)
            .arg(0u32)
            .arg(0u32)
            .arg(0u32)
            .build()?;
        Ok(kernel)
    }

    fn make_context() -> Result<ocl::Context, Error> {
        let mut last_err = None;
        let selected = ::std::env::var("CLAM5_DEVICE").ok();
        let mut i = 0;
        println!("Devices (pass env var CLAM5_DEVICE={{i}} to select)");
        for platform in ocl::Platform::list() {
            println!("Platform: {}", platform.name()?);
            for device in ocl::Device::list(platform, None)? {
                match selected {
                    Some(ref selected) if selected.parse() == Ok(i) => {
                        return Ok(ocl::Context::builder()
                            .platform(platform)
                            .devices(device)
                            .build()?)
                    }
                    Some(ref selected) if selected.parse::<i32>().is_ok() => (),
                    _ => println!("[{}]: {}", i, device.name()?),
                }
                i += 1;
            }
        }

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

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        self.width = width;
        self.height = height;
        self.data = None;
        self.frame = 0;
        Ok(())
    }

    fn set_args(&mut self, settings: &Settings) -> Result<(), Error> {
        let old_cfg = self.cpu_cfg;
        self.cpu_cfg.read(settings);
        if old_cfg != self.cpu_cfg {
            let to_write = [self.cpu_cfg];
            self.cfg.write(&to_write as &[_]).queue(&self.queue).enq()?;
            self.frame = 0;
        }
        if *settings != self.old_settings {
            self.old_settings = settings.clone();
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
        let render_scale = (*settings.get_u32("render_scale").unwrap()).max(1);
        let width = self.width / render_scale;
        let height = self.height / render_scale;
        self.kernel.set_arg(0, data)?;
        self.kernel.set_arg(1, &self.cfg)?;
        self.kernel.set_arg(2, width as u32)?;
        self.kernel.set_arg(3, height as u32)?;
        self.kernel.set_arg(4, self.frame as u32)?;
        Ok(())
    }

    pub fn run(&mut self, settings: &Settings, download: bool) -> Result<Option<Image>, Error> {
        self.set_args(settings)?;
        let lws = 1024;
        let render_scale = (*settings.get_u32("render_scale").unwrap()).max(1);
        let width = self.width / render_scale;
        let height = self.height / render_scale;
        let total_size = width * height;
        let to_launch = self
            .kernel
            .cmd()
            .queue(&self.queue)
            .global_work_size((total_size + lws - 1) / lws * lws);
        // enq() is unsafe, even though the Rust code is safe (unsafe due to untrusted GPU code)
        unsafe { to_launch.enq() }?;
        self.frame += 1;
        if download {
            let mut vec = vec![0u8; width as usize * height as usize * 4];
            self.data
                .as_ref()
                .unwrap()
                .read(&mut vec)
                .queue(&self.queue)
                .enq()?;
            let image = Image::new(vec, width, height);
            Ok(Some(image))
        } else {
            Ok(None)
        }
    }

    pub fn sync(&mut self) -> ocl::Result<()> {
        self.queue.finish()
    }
}
