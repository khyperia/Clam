use display::Image;
use failure;
use failure::Error;
use kernel_compilation;
use ocl;
use ocl::enums::ContextPropertyValue;
use settings::Settings;

const DATA_WORDS: u32 = 6;

struct KernelImage {
    queue: ocl::Queue,
    width: u32,
    height: u32,
    scale: u32,
    buffer: Option<ocl::Buffer<u8>>,
    buffer_gl: Option<ocl::prm::cl_GLuint>,
}

impl KernelImage {
    fn new(queue: ocl::Queue, width: u32, height: u32) -> Self {
        Self {
            queue,
            width,
            height,
            scale: 1,
            buffer: None,
            buffer_gl: None,
        }
    }

    fn size(&self) -> (u32, u32) {
        (self.width / self.scale, self.height / self.scale)
    }

    fn test(&mut self) {
        let mut texture = 0;
        let width = 200;
        let height = 200;
        let array_size = width * height * 4;
        let row_pitch = width * 4;
        let slc_pitch = width * height * 4;
        unsafe {
            let () = gl::CreateTextures(gl::TEXTURE_2D, 1, &mut texture);
            ::check_gl().unwrap();
            gl::TextureStorage2D(texture, 1, gl::RGBA8, 200, 200);
            ::check_gl().unwrap();
        }
        let buffer_cl: ocl::Image<u8> = ocl::Image::from_gl_texture(
            &self.queue,
            ocl::flags::MemFlags::new().read_write(),
            ocl::builders::ImageDescriptor::new(
                ocl::enums::MemObjectType::Image2d,
                width,
                height,
                1,
                array_size,
                row_pitch,
                slc_pitch,
                None,
            ),
            ocl::core::GlTextureTarget::GlTexture2d,
            0,
            texture,
        )
        .unwrap();
    }

    fn data(&mut self, is_ogl: bool) -> Result<&ocl::Buffer<u8>, Error> {
        if self.buffer.is_none() {
            if is_ogl {
                //self.test();
                unsafe {
                    ::check_gl()?;
                    let mut buffer = 0;
                    let () = gl::CreateBuffers(1, &mut buffer);
                    ::check_gl()?;
                    gl::NamedBufferData(
                        buffer,
                        self.width as gl::types::GLsizeiptr
                            * self.height as gl::types::GLsizeiptr
                            * DATA_WORDS as gl::types::GLsizeiptr
                            * 4,
                        std::ptr::null(),
                        gl::STREAM_DRAW, // TODO: fix this
                    );
                    self.buffer_gl = Some(buffer);
                    let buffer_cl = ocl::Buffer::from_gl_buffer(&self.queue, None, buffer)?;
                    self.buffer = Some(buffer_cl);
                }
            } else {
                let new_data = ocl::Buffer::builder()
                    .context(&self.queue.context())
                    .len(self.width * self.height * DATA_WORDS * 4)
                    .build()?;
                self.buffer = Some(new_data);
            }
        }
        Ok(self.buffer.as_ref().unwrap())
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        if self.width != new_width || self.height != new_height {
            self.width = new_width;
            self.height = new_height;
            self.buffer = None;
            if let Some(id) = self.buffer_gl {
                unsafe {
                    gl::DeleteBuffers(1, &id);
                }
            }
            self.buffer_gl = None;
        }
    }

    fn rescale(&mut self, new_scale: u32) {
        self.scale = new_scale.max(1);
    }

    fn download(&mut self) -> Result<Image, Error> {
        let (width, height) = self.size();

        if let Some(gl) = self.buffer_gl {
            Ok(Image::new(None, Some(gl), width, height))
        } else {
            let mut vec = vec![0u8; width as usize * height as usize * 4];

            // only read if data is present
            if let Some(ref buffer) = self.buffer {
                buffer.read(&mut vec).queue(&self.queue).enq()?;
            }

            Ok(Image::new(Some(vec), None, width, height))
        }
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
    is_ogl: bool,
}

impl Kernel {
    pub fn create(
        width: u32,
        height: u32,
        is_ogl: bool,
        window: Option<(&sdl2::video::Window, &sdl2::video::GLContext)>,
        settings: &mut Settings,
    ) -> Result<Self, Error> {
        // TODO: lazy_static context
        let context = Self::make_context(window)?;
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
            is_ogl,
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

    fn make_context(
        window: Option<(&sdl2::video::Window, &sdl2::video::GLContext)>,
    ) -> Result<ocl::Context, Error> {
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
                        return Self::build_ctx(platform, Some(device), window, false);
                    }
                    Some(_) => (),
                    None => println!("[{}]: {}", i, device.name()?),
                }
                i += 1;
            }
        }

        for platform in ocl::Platform::list() {
            match Self::build_ctx(platform, None, window, true) {
                Ok(ok) => return Ok(ok),
                Err(e) => last_err = Some(e),
            }
        }

        for platform in ocl::Platform::list() {
            match Self::build_ctx(platform, None, window, false) {
                Ok(ok) => return Ok(ok),
                Err(e) => last_err = Some(e),
            }
        }

        match last_err {
            Some(e) => Err(e.into()),
            None => Err(failure::err_msg("No OpenCL devices found")),
        }
    }

    fn build_ctx(
        platform: ocl::Platform,
        device: Option<ocl::Device>,
        window: Option<(&sdl2::video::Window, &sdl2::video::GLContext)>,
        gpu: bool,
    ) -> Result<ocl::Context, Error> {
        let mut builder = ocl::Context::builder();
        builder.platform(platform);
        if let Some(device) = device {
            builder.devices(device);
        }
        if gpu && false {
            builder.devices(ocl::DeviceType::new().gpu());
        }
        if let Some((sdl_window, sdl_context)) = window {
            unsafe {
                let wglGetCurrentContext: extern "system" fn() -> *mut libc::c_void =
                    std::mem::transmute(
                        sdl_window
                            .subsystem()
                            .gl_get_proc_address("wglGetCurrentContext"),
                    );
                let wglGetCurrentDC: extern "system" fn() -> *mut libc::c_void =
                    std::mem::transmute(
                        sdl_window
                            .subsystem()
                            .gl_get_proc_address("wglGetCurrentDC"),
                    );
                println!(" from sdl cont {:?}", sdl_context.raw());
                println!(" winwin {:?}", sdl_window.raw());
                let wglCurrentContext = wglGetCurrentContext();
                let wglCurrentDC = wglGetCurrentDC();
                println!("{:?} {:?}", wglCurrentContext, wglCurrentDC);
                builder.property(ContextPropertyValue::GlContextKhr(wglCurrentContext));
                builder.property(ContextPropertyValue::WglHdcKhr(wglCurrentDC));
            }
            //builder.property(ContextPropertyValue::GlContextKhr());
        }
        return Ok(builder.build()?);
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
            kernel.set_arg(0, self.data.data(self.is_ogl)?)?;
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

            if self.is_ogl {
                if let Some(ref buf) = self.data.buffer {
                    buf.cmd().gl_acquire().enq()?;
                }
            }
            let to_launch = kernel.cmd().queue(&self.queue).global_work_size(total_size);
            // enq() is unsafe, even though the Rust code is safe (unsafe due to untrusted GPU code)
            unsafe { to_launch.enq() }?;

            if self.is_ogl {
                if let Some(ref buf) = self.data.buffer {
                    buf.cmd().gl_release().enq()?;
                }
            }

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
