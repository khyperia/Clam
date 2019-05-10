use display::Image;
use failure;
use failure::Error;
use kernel_compilation;
use ocl;
use settings::Settings;

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
    kernel: Option<ocl::Kernel>,
    data: KernelImage,
    cpu_cfg: Vec<u8>,
    cfg: Option<ocl::Buffer<u8>>,
    old_settings: Settings,
    frame: u32,
}

impl Kernel {
    pub fn create(width: u32, height: u32, settings: &mut Settings) -> Result<Self, Error> {
        // if !kernel_compilation::check_header(settings) {
        //     return Err(failure::err_msg("Header didn't start with proper struct"));
        // }
        let context = Self::make_context()?;
        let device = context.devices()[0];
        let device_name = device.name()?;
        println!("Using device: {}", device_name);
        let queue = ocl::Queue::new(&context, device, None)?;
        let mut result = Kernel {
            queue: queue.clone(),
            kernel: None,
            data: KernelImage::new(queue, width, height),
            cpu_cfg: Vec::new(),
            cfg: None,
            old_settings: settings.clone(),
            frame: 0,
        };
        result.rebuild(settings)?;
        Ok(result)
    }

    pub fn rebuild(&mut self, settings: &mut Settings) -> Result<(), Error> {
        let new_kernel = kernel_compilation::rebuild(&self.queue, settings);
        match new_kernel {
            Ok(k) => {
                self.kernel = Some(k);
                self.frame = 0;
            }
            Err(err) => println!("Kernel compilation failed: {}", err),
        }
        Ok(())
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
        let new_cfg = settings.serialize();
        if new_cfg != self.cpu_cfg {
            if new_cfg.len() != self.cpu_cfg.len() {
                if new_cfg.is_empty() {
                    self.cfg = None;
                } else {
                    self.cfg = Some(
                        ocl::Buffer::builder()
                            .context(&self.queue.context())
                            .len(new_cfg.len())
                            .build()?,
                    );
                }
            }
            self.cpu_cfg = new_cfg;
            let to_write = &self.cpu_cfg as &[_];
            if let Some(ref cfg) = self.cfg {
                cfg.write(to_write).queue(&self.queue).enq()?;
            }
            self.frame = 0;
        }

        if *settings != self.old_settings {
            self.old_settings = settings.clone();
            self.frame = 0;
        }

        self.data
            .rescale(settings.find("render_scale").unwrap_u32());

        Ok(())
    }

    fn set_args(&mut self) -> Result<(), Error> {
        if let Some(ref kernel) = self.kernel {
            let (width, height) = self.data.size();
            kernel.set_arg(0, self.data.data()?)?;
            kernel.set_arg(1, self.cfg.as_ref())?;
            kernel.set_arg(2, width as u32)?;
            kernel.set_arg(3, height as u32)?;
            kernel.set_arg(4, self.frame as u32)?;
        }
        Ok(())
    }

    fn launch(&mut self) -> Result<(), Error> {
        if let Some(ref kernel) = self.kernel {
            let (width, height) = self.data.size();
            let total_size = width * height;

            let to_launch = kernel.cmd().queue(&self.queue).global_work_size(total_size);
            // enq() is unsafe, even though the Rust code is safe (unsafe due to untrusted GPU code)
            unsafe { to_launch.enq() }?;

            self.frame += 1;
        }

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
