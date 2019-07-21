use crate::check_gl;
use crate::interactive::ImageData;
use crate::kernel_compilation;
use crate::settings::Settings;
use failure;
use failure::Error;
use gl::types::*;
use ocl::builders::ContextBuilder;
use ocl::builders::ImageDescriptor;
use ocl::builders::ImageFormat;
use ocl::core::GlTextureTarget;
use ocl::enums::ContextPropertyValue;
use ocl::enums::ImageChannelDataType;
use ocl::enums::ImageChannelOrder;
use ocl::enums::MemObjectType;
use ocl::flags::MemFlags;
use ocl::prm::cl_GLuint;
use ocl::Buffer;
use ocl::Context;
use ocl::Device;
use ocl::DeviceType;
use ocl::Image;
use ocl::Kernel;
use ocl::OclPrm;
use ocl::Platform;
use ocl::Queue;
use std::ffi::c_void;

// SSSRR
const DATA_WORDS: u32 = 5;

struct KernelImage<T: OclPrm> {
    queue: Queue,
    width: u32,
    height: u32,
    scale: u32,
    scratch: Option<Buffer<f32>>,
    texture_cl: Option<Image<T>>,
    texture_gl: Option<cl_GLuint>,
}

impl<T: OclPrm> KernelImage<T> {
    fn new(queue: Queue, width: u32, height: u32) -> Self {
        Self {
            queue,
            width,
            height,
            scale: 1,
            scratch: None,
            texture_cl: None,
            texture_gl: None,
        }
    }

    fn size(&self) -> (u32, u32) {
        (self.width / self.scale, self.height / self.scale)
    }

    fn data(&mut self, is_ogl: bool) -> Result<(&Image<T>, &Buffer<f32>), Error> {
        let (width, height) = self.size();
        if self.texture_cl.is_none() {
            unsafe {
                if is_ogl {
                    let (cl, gl) = cl_gl_buf(&self.queue, width as _, height as _)?;
                    self.texture_cl = Some(cl);
                    self.texture_gl = Some(gl);
                } else {
                    let cl = cl_buf(&self.queue, width as _, height as _)?;
                    self.texture_cl = Some(cl);
                }
            }
        }
        if self.scratch.is_none() {
            let new_data = Buffer::builder()
                .context(&self.queue.context())
                .len(width * height * DATA_WORDS)
                .build()?;
            self.scratch = Some(new_data);
        }
        Ok((
            self.texture_cl.as_ref().unwrap(),
            self.scratch.as_ref().unwrap(),
        ))
    }

    fn resize(&mut self, new_width: u32, new_height: u32, new_scale: u32) {
        let old_size = self.size();
        self.width = new_width;
        self.height = new_height;
        self.scale = new_scale.max(1);
        if old_size != self.size() {
            self.scratch = None;
            self.texture_cl = None;
            if let Some(id) = self.texture_gl {
                unsafe {
                    gl::DeleteBuffers(1, &id);
                }
            }
            self.texture_gl = None;
        }
    }

    fn download(&mut self) -> Result<ImageData<T>, Error> {
        let (width, height) = self.size();

        if let Some(gl) = self.texture_gl {
            Ok(ImageData::new(None, Some(gl), width, height))
        } else {
            let mut vec = vec![T::default(); width as usize * height as usize * 4];

            // only read if data is present
            if let Some(ref buffer) = self.texture_cl {
                buffer
                    .read(&mut vec)
                    .queue(&self.queue)
                    .region([width, height, 1])
                    .enq()?;
            }

            Ok(ImageData::new(Some(vec), None, width, height))
        }
    }
}

unsafe fn cl_gl_buf<T: OclPrm>(
    queue: &Queue,
    width: usize,
    height: usize,
) -> Result<(Image<T>, cl_GLuint), Error> {
    let mut texture = 0;
    gl::CreateTextures(gl::TEXTURE_2D, 1, &mut texture);
    check_gl()?;
    if std::mem::size_of::<T>() == 1 {
        gl::TextureStorage2D(texture, 1, gl::RGBA8UI, width as _, height as _);
    } else if std::mem::size_of::<T>() == 4 {
        gl::TextureStorage2D(texture, 1, gl::RGBA32F, width as _, height as _);
    } else {
        panic!("std::mem::size_of::<T>() did not equal 1 or 4");
    }
    check_gl()?;
    gl::TextureParameteri(texture, gl::TEXTURE_MIN_FILTER, gl::NEAREST as GLint);
    check_gl()?;
    gl::TextureParameteri(texture, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint);
    check_gl()?;

    let image = Image::from_gl_texture(
        queue,
        MemFlags::new().write_only(),
        ImageDescriptor::new(MemObjectType::Image2d, width, height, 0, 0, 0, 0, None),
        GlTextureTarget::GlTexture2d,
        0,
        texture,
    )?;

    check_gl()?;

    Ok((image, texture))
}

unsafe fn cl_buf<T: OclPrm>(queue: &Queue, width: usize, height: usize) -> Result<Image<T>, Error> {
    let data_type = if std::mem::size_of::<T>() == 1 {
        ImageChannelDataType::UnsignedInt8
    } else if std::mem::size_of::<T>() == 4 {
        ImageChannelDataType::Float
    } else {
        panic!("std::mem::size_of::<T>() did not equal 1 or 4");
    };
    let cl = Image::new(
        queue,
        MemFlags::new().read_write(),
        ImageFormat::new(ImageChannelOrder::Rgba, data_type),
        ImageDescriptor::new(MemObjectType::Image2d, width, height, 0, 0, 0, 0, None),
        None,
    )?;
    Ok(cl)
}

pub struct FractalKernel<T: OclPrm> {
    queue: Queue,
    kernel: Option<Kernel>,
    data: KernelImage<T>,
    cpu_cfg: Vec<u8>,
    cfg: Option<Buffer<u8>>,
    old_settings: Settings,
    frame: u32,
    is_ogl: bool,
}

impl<T: OclPrm> FractalKernel<T> {
    pub fn create(
        width: u32,
        height: u32,
        is_ogl: bool,
        settings: &mut Settings,
    ) -> Result<Self, Error> {
        lazy_static! {
            static ref CONTEXT: Context = make_context().expect("Could not create OpenCL context");
        }
        let device = CONTEXT.devices()[0];
        let device_name = device.name()?;
        println!("Using device: {}", device_name);
        let queue = Queue::new(&CONTEXT, device, None)?;
        let mut result = Self {
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
        let (texture, _) = self.data.data(self.is_ogl)?;
        let new_kernel = kernel_compilation::rebuild(&self.queue, texture, settings);
        match new_kernel {
            Ok(k) => {
                self.kernel = Some(k);
                self.frame = 0;
            }
            //Err(err) => println!("Kernel compilation failed: {}", err),
            Err(err) => return Err(err), // TODO
        }
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        self.data.resize(width, height, self.data.scale);
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
                        Buffer::builder()
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

        self.data.resize(
            self.data.width,
            self.data.height,
            settings.find("render_scale").unwrap_u32(),
        );

        Ok(())
    }

    fn set_args(&mut self) -> Result<(), Error> {
        if let Some(ref kernel) = self.kernel {
            let (width, height) = self.data.size();
            let (texture, scratch) = self.data.data(self.is_ogl)?;
            kernel.set_arg(0, texture)?;
            kernel.set_arg(1, scratch)?;
            kernel.set_arg(2, self.cfg.as_ref())?;
            kernel.set_arg(3, width as u32)?;
            kernel.set_arg(4, height as u32)?;
            kernel.set_arg(5, self.frame as u32)?;
        }
        Ok(())
    }

    fn launch(&mut self) -> Result<(), Error> {
        if let Some(ref kernel) = self.kernel {
            let (width, height) = self.data.size();
            let total_size = width * height;

            if self.is_ogl {
                if let Some(ref buf) = self.data.texture_cl {
                    buf.cmd().gl_acquire().enq()?;
                }
            }
            let to_launch = kernel.cmd().queue(&self.queue).global_work_size(total_size);
            // enq() is unsafe, even though the Rust code is safe (unsafe due to untrusted GPU code)
            unsafe { to_launch.enq() }?;

            if self.is_ogl {
                if let Some(ref buf) = self.data.texture_cl {
                    buf.cmd().gl_release().enq()?;
                }
            }

            self.frame += 1;
        }

        Ok(())
    }

    pub fn run(&mut self, settings: &mut Settings, force_rebuild: bool) -> Result<(), Error> {
        if settings.check_rebuild() || force_rebuild {
            self.rebuild(settings)?;
            println!("Rebuilding");
        }
        self.update(settings)?;
        self.set_args()?;
        self.launch()?;
        Ok(())
    }

    pub fn download(&mut self) -> Result<ImageData<T>, Error> {
        self.data.download()
    }

    pub fn sync_renderer(&mut self) -> Result<(), Error> {
        Ok(self.queue.finish()?)
    }
}

fn make_context() -> Result<Context, Error> {
    let mut last_err = None;
    let selected = ::std::env::var("CLAM5_DEVICE")
        .ok()
        .and_then(|x| x.parse::<u32>().ok());
    let mut i = 0;
    println!("Devices (pass env var CLAM5_DEVICE={{i}} to select)");
    for platform in Platform::list() {
        println!(
            "Platform: {} (version: {})",
            platform.name()?,
            platform.version()?,
        );
        for device in Device::list(platform, None)? {
            match selected {
                Some(selected) if selected == i => {
                    return build_ctx(platform, Some(device), false);
                }
                Some(_) => (),
                None => println!("[{}]: {}", i, device.name()?),
            }
            i += 1;
        }
    }

    for platform in Platform::list() {
        match build_ctx(platform, None, true) {
            Ok(ok) => return Ok(ok),
            Err(e) => last_err = Some(e),
        }
    }

    for platform in Platform::list() {
        match build_ctx(platform, None, false) {
            Ok(ok) => return Ok(ok),
            Err(e) => last_err = Some(e),
        }
    }

    match last_err {
        Some(e) => Err(e),
        None => Err(failure::err_msg("No OpenCL devices found")),
    }
}

extern "system" fn dummy() -> *mut c_void {
    std::ptr::null_mut()
}
static mut WGL_GET_CURRENT_CONTEXT: extern "system" fn() -> *mut c_void = dummy;
static mut WGL_GET_CURRENT_DC: extern "system" fn() -> *mut c_void = dummy;
static mut GLX_GET_CURRENT_CONTEXT: extern "system" fn() -> *mut c_void = dummy;
static mut GLX_GET_CURRENT_DISPLAY: extern "system" fn() -> *mut c_void = dummy;

pub fn init_gl_funcs(video: &sdl2::VideoSubsystem) {
    unsafe fn init(
        video: &sdl2::VideoSubsystem,
        func: &mut extern "system" fn() -> *mut c_void,
        name: &str,
    ) {
        let addr = video.gl_get_proc_address(name);
        if !addr.is_null() {
            println!("Have {}: {:?}", name, addr);
            *func = std::mem::transmute(addr);
        }
    }

    unsafe {
        // Note: glXGetProcAddress returns valid pointers, even for invalid strings.
        // https://dri.freedesktop.org/wiki/glXGetProcAddressNeverReturnsNULL/
        init(video, &mut WGL_GET_CURRENT_CONTEXT, "wglGetCurrentContext");
        init(video, &mut WGL_GET_CURRENT_DC, "wglGetCurrentDC");
        init(video, &mut GLX_GET_CURRENT_CONTEXT, "glXGetCurrentContext");
        init(video, &mut GLX_GET_CURRENT_DISPLAY, "glXGetCurrentDisplay");
    }
}

fn build_ctx(platform: Platform, device: Option<Device>, gpu: bool) -> Result<Context, Error> {
    let mut builder = Context::builder();
    builder.platform(platform);
    if let Some(device) = device {
        builder.devices(device);
    }
    // TODO: Check if !is_ogl is needed here
    if gpu {
        builder.devices(DeviceType::new().gpu());
    }
    let wgl = cfg!(windows) && build_ctx_wgl(&mut builder);
    let glx = !wgl && cfg!(not(windows)) && build_ctx_glx(&mut builder);
    if !wgl && !glx {
        println!("No OpenGL context found");
    }
    Ok(builder.build()?)
}

fn build_ctx_wgl(builder: &mut ContextBuilder) -> bool {
    let gl_context_khr = unsafe { WGL_GET_CURRENT_CONTEXT }();
    let wgl_hdc_khr = unsafe { WGL_GET_CURRENT_DC }();
    if !gl_context_khr.is_null() && !wgl_hdc_khr.is_null() {
        builder.property(ContextPropertyValue::GlContextKhr(gl_context_khr));
        builder.property(ContextPropertyValue::WglHdcKhr(wgl_hdc_khr));
        println!("WGL OpenCL");
        true
    } else {
        false
    }
}

fn build_ctx_glx(builder: &mut ContextBuilder) -> bool {
    let gl_context_khr = unsafe { GLX_GET_CURRENT_CONTEXT }();
    let glx_display_khr = unsafe { GLX_GET_CURRENT_DISPLAY }();
    if !gl_context_khr.is_null() && !glx_display_khr.is_null() {
        builder.property(ContextPropertyValue::GlContextKhr(gl_context_khr));
        builder.property(ContextPropertyValue::GlxDisplayKhr(glx_display_khr));
        println!("GLX OpenCL");
        true
    } else {
        false
    }
}
