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
const DATA_WORDS: u32 = 6;

struct KernelImage {
    queue: ocl::Queue,
    width: u32,
    height: u32,
    scale: u32,
    buffer: Option<ocl::Buffer<u8>>,
}

impl KernelImage {
    fn new(queue: ocl::Queue, width: u32, height: u32) -> Self {
        Self {
            queue,
            width,
            height,
            scale: 1,
            buffer: None,
        }
    }

    fn size(&self) -> (u32, u32) {
        (self.width / self.scale, self.height / self.scale)
    }

    fn data(&mut self) -> Result<&ocl::Buffer<u8>, Error> {
        if self.buffer.is_none() {
            let new_data = ocl::Buffer::builder()
                .context(&self.queue.context())
                .len(self.width * self.height * DATA_WORDS * 4)
                .build()?;
            self.buffer = Some(new_data);
        }
        Ok(self.buffer.as_ref().unwrap())
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        if self.width != new_width || self.height != new_height {
            self.width = new_width;
            self.height = new_height;
            self.buffer = None;
        }
    }

    fn rescale(&mut self, new_scale: u32) {
        self.scale = new_scale.max(1);
    }

    fn download(&mut self) -> Result<Image, Error> {
        let (width, height) = self.size();
        let mut vec = vec![0u8; width as usize * height as usize * 4];

        // only read if data is present
        if let Some(ref buffer) = self.buffer {
            buffer.read(&mut vec).queue(&self.queue).enq()?;
        }

        //self.queue.finish()?;

        let image = Image::new(vec, width, height);
        Ok(image)
    }
}

pub struct Kernel {
    queue: ocl::Queue,
    kernel: ocl::Kernel,
    data: KernelImage,
    cpu_cfg: MandelboxCfg,
    cfg: ocl::Buffer<MandelboxCfg>,
    old_settings: Settings,
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

fn generate_rust_struct(settings: &Settings) -> String {
    let mut result = String::new();
    result.push_str("pub struct MandelboxCfg {\n");
    for name in Settings::keys() {
        if name == "render_scale" {
            continue;
        }
        let value = settings.get(name).unwrap();
        match value {
            SettingValue::U32(_) => result.push_str(&format!("    {}: u32,\n", name)),
            SettingValue::F32(_, _) => result.push_str(&format!("    {}: f32,\n", name)),
        }
    }
    result.push_str("}\n");
    result
}

fn generate_header(settings: &Settings) -> String {
    let mut result = String::new();
    result.push_str("struct MandelboxCfg\n");
    result.push_str("{\n");
    for name in Settings::keys() {
        if name == "render_scale" {
            continue;
        }
        let value = settings.get(name).unwrap();
        match value {
            SettingValue::U32(_) => result.push_str(&format!("    int _{};\n", name)),
            SettingValue::F32(_, _) => result.push_str(&format!("    float _{};\n", name)),
        }
    }
    result.push_str("};\n\n");
    for name in Settings::keys() {
        if name == "render_scale" {
            continue;
        }
        result.push_str(&format!("#ifndef {}\n", name));
        result.push_str(&format!("#define {} cfg->_{}\n", name, name));
        result.push_str(&format!("#endif\n"));
    }
    result
}

fn check_header(settings: &Settings) -> bool {
    let header = generate_header(settings);
    if !MANDELBOX.starts_with(&header) {
        println!("Header check failed:");
        println!("{}", header);
        println!("-----");
        println!("{}", generate_rust_struct(settings));
        false
    } else {
        true
    }
}

impl Kernel {
    pub fn create(width: u32, height: u32, settings: &Settings) -> Result<Self, Error> {
        if !check_header(settings) {
            return Err(failure::err_msg("Header didn't start with proper struct"));
        }
        let context = Self::make_context()?;
        let device = context.devices()[0];
        let device_name = device.name()?;
        println!("Using device: {}", device_name);
        let queue = ocl::Queue::new(&context, device, None)?;
        let kernel = Self::rebuild(&queue, settings)?;
        let cfg = ocl::Buffer::builder().context(&context).len(1).build()?;
        Ok(Kernel {
            queue: queue.clone(),
            kernel,
            data: KernelImage::new(queue, width, height),
            cpu_cfg: MandelboxCfg::default(),
            cfg,
            old_settings: settings.clone(),
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
        let selected = ::std::env::var("CLAM5_DEVICE")
            .ok()
            .and_then(|x| x.parse::<u32>().ok());
        let mut i = 0;
        println!("Devices (pass env var CLAM5_DEVICE={{i}} to select)");
        for platform in ocl::Platform::list() {
            println!(
                "Platform: {} (version: {})",
                platform.name()?,
                platform.version()?,
            );
            for device in ocl::Device::list(platform, None)? {
                match selected {
                    Some(selected) if selected == i => {
                        return Ok(ocl::Context::builder()
                            .platform(platform)
                            .devices(device)
                            .build()?);
                    }
                    Some(_) => (),
                    None => println!("[{}]: {}", i, device.name()?),
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
        self.data.resize(width, height);
        self.frame = 0;
        Ok(())
    }

    fn update(&mut self, settings: &Settings) -> Result<(), Error> {
        let old_cfg = self.cpu_cfg;
        self.cpu_cfg.read(settings);
        if old_cfg != self.cpu_cfg {
            let to_write = &[self.cpu_cfg] as &[_];
            self.cfg.write(to_write).queue(&self.queue).enq()?;
            self.frame = 0;
        }

        if *settings != self.old_settings {
            self.old_settings = settings.clone();
            self.frame = 0;
        }

        if let Some(&SettingValue::U32(render_scale)) = settings.get("render_scale") {
            self.data.rescale(render_scale);
        }

        Ok(())
    }

    fn set_args(&mut self) -> Result<(), Error> {
        let (width, height) = self.data.size();
        self.kernel.set_arg(0, self.data.data()?)?;
        self.kernel.set_arg(1, &self.cfg)?;
        self.kernel.set_arg(2, width as u32)?;
        self.kernel.set_arg(3, height as u32)?;
        self.kernel.set_arg(4, self.frame as u32)?;
        Ok(())
    }

    fn launch(&mut self) -> Result<(), Error> {
        let (width, height) = self.data.size();
        let total_size = width * height;

        let to_launch = self
            .kernel
            .cmd()
            .queue(&self.queue)
            .global_work_size(total_size);
        // enq() is unsafe, even though the Rust code is safe (unsafe due to untrusted GPU code)
        unsafe { to_launch.enq() }?;

        self.frame += 1;

        Ok(())
    }

    pub fn run(&mut self, settings: &Settings) -> Result<(), Error> {
        self.update(settings)?;
        self.set_args()?;
        self.launch()?;
        Ok(())
    }

    pub fn download(&mut self) -> Result<Image, Error> {
        self.data.download()
    }

    pub fn sync_renderer(&mut self) -> ocl::Result<()> {
        self.queue.finish()
    }
}
