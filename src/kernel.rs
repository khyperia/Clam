use crate::{
    check_gl,
    kernel_compilation::{ComputeShader, RealizedSource},
    settings::Settings,
    Error,
};
use khygl::texture::{CpuTexture, Texture, TextureType};

struct KernelImage<T: TextureType> {
    width: usize,
    height: usize,
    scale: usize,
    output: Texture<T>,
    scratch: Texture<[f32; 4]>,
    randbuf: Texture<u32>,
}

impl<T: TextureType> KernelImage<T> {
    fn new(width: usize, height: usize) -> Result<Self, Error> {
        Ok(Self {
            width,
            height,
            scale: 1,
            output: Texture::new((width, height))?,
            scratch: Texture::new((width, height))?,
            randbuf: Texture::new((width, height))?,
        })
    }

    fn size(&self) -> (usize, usize) {
        (self.width / self.scale, self.height / self.scale)
    }

    fn resize(
        &mut self,
        new_width: usize,
        new_height: usize,
        new_scale: usize,
    ) -> Result<bool, Error> {
        let old_size = self.size();
        self.width = new_width;
        self.height = new_height;
        self.scale = new_scale.max(1);
        let new_size = self.size();
        if old_size != new_size {
            self.output = Texture::new(new_size)?;
            self.scratch = Texture::new(new_size)?;
            self.randbuf = Texture::new(new_size)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

pub struct Kernel<T: TextureType> {
    kernel: Option<ComputeShader>,
    data: KernelImage<T>,
    old_settings: Settings,
    realized_source: RealizedSource,
    local_size: usize,
    frame: u32,
}

impl<T: TextureType> Kernel<T> {
    pub fn create(
        realized_source: RealizedSource,
        width: usize,
        height: usize,
    ) -> Result<Self, Error> {
        let mut local_size = 0;
        unsafe {
            gl::GetIntegeri_v(gl::MAX_COMPUTE_WORK_GROUP_SIZE, 0, &mut local_size);
            check_gl()?;
        }
        local_size = local_size.min(64);
        let result = Self {
            kernel: None,
            data: KernelImage::new(width, height)?,
            old_settings: Settings::new(),
            realized_source,
            local_size: local_size as usize,
            frame: 0,
        };
        Ok(result)
    }

    pub fn set_src(&mut self, src: RealizedSource) {
        self.realized_source = src;
        self.kernel = None;
    }

    pub fn realized_source(&self) -> &RealizedSource {
        &self.realized_source
    }

    fn try_rebuild(&mut self, settings: &Settings) -> Result<(), Error> {
        if settings.check_rebuild(&self.old_settings) || self.kernel.is_none() {
            println!("Rebuilding");
            self.old_settings = Settings::new();
            let new_kernel = self.realized_source.rebuild(settings, self.local_size);
            match new_kernel {
                Ok(k) => {
                    self.kernel = Some(k);
                    self.frame = 0;
                }
                //Err(err) => println!("Kernel compilation failed: {}", err),
                Err(err) => return Err(err), // TODO
            }
        }
        Ok(())
    }

    pub fn resize(&mut self, width: usize, height: usize) -> Result<(), Error> {
        if self.data.resize(width, height, self.data.scale)? {
            self.frame = 0;
        }
        Ok(())
    }

    fn update(&mut self, settings: &Settings) -> Result<(), Error> {
        if self.data.resize(
            self.data.width,
            self.data.height,
            settings.find("render_scale").unwrap_u32() as usize,
        )? {
            self.frame = 0;
        }

        if let Some(kernel) = &self.kernel {
            let mut frame = None;
            let (width, height) = self.data.size();
            for uniform in kernel.uniforms() {
                match (&uniform.name as &str, uniform.ty) {
                    ("width", _) => uniform.set_arg_u32(width as u32)?,
                    ("height", _) => uniform.set_arg_u32(height as u32)?,
                    ("frame", _) => frame = Some(uniform),
                    (_, gl::IMAGE_2D) | (_, gl::UNSIGNED_INT_IMAGE_2D) => (),
                    (name, _) => {
                        let old = self.old_settings.get(&name).map(|v| v.value());
                        let new = settings.find(name).value();
                        if old != Some(new) {
                            new.set_uniform(uniform)?;
                            self.frame = 0;
                        }
                    }
                }
            }
            // delay setting frame to get the reset if another variable changes
            if let Some(frame) = frame {
                frame.set_arg_u32(self.frame)?;
            }
        }

        self.old_settings = settings.clone();

        Ok(())
    }

    fn launch(&mut self) -> Result<(), Error> {
        if let Some(kernel) = &self.kernel {
            let (width, height) = self.data.size();
            let total_size = width * height;
            let local_size = self.local_size;
            let launch_size = (total_size + local_size - 1) / local_size;
            unsafe {
                gl::UseProgram(kernel.shader);
                check_gl()?;
                self.data.output.bind(0)?;
                self.data.scratch.bind(1)?;
                self.data.randbuf.bind(2)?;
                check_gl()?;
                gl::DispatchCompute(launch_size as u32, 1, 1);
                check_gl()?;
                gl::UseProgram(0);
                check_gl()?;
            }
            self.frame += 1;
        }
        Ok(())
    }

    pub fn run(&mut self, settings: &Settings) -> Result<(), Error> {
        self.try_rebuild(settings)?;
        self.update(settings)?;
        self.launch()?;
        Ok(())
    }

    pub fn texture(&mut self) -> &Texture<T> {
        &self.data.output
    }

    pub fn download(&mut self) -> Result<CpuTexture<T>, Error> {
        self.data.output.download()
    }

    pub fn sync_renderer(&mut self) -> Result<(), Error> {
        unsafe {
            gl::Finish();
            check_gl()
        }
    }
}
