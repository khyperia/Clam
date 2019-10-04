use crate::{
    check_gl,
    gl_help::{set_arg_u32, CpuTexture, Texture, TextureType},
    kernel_compilation,
    settings::Settings,
};
use failure::{self, Error};
use gl::types::*;

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
            output: Texture::new(width, height)?,
            scratch: Texture::new(width, height)?,
            randbuf: Texture::new(width, height)?,
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
    ) -> Result<(), Error> {
        let old_size = self.size();
        self.width = new_width;
        self.height = new_height;
        self.scale = new_scale.max(1);
        let new_size = self.size();
        if old_size != new_size {
            self.output = Texture::new(new_size.0, new_size.1)?;
            self.scratch = Texture::new(new_size.0, new_size.1)?;
            self.randbuf = Texture::new(new_size.0, new_size.1)?;
        }
        Ok(())
    }
}

pub struct Kernel<T: TextureType> {
    kernel: Option<GLuint>,
    data: KernelImage<T>,
    old_settings: Settings,
    frame: u32,
    local_size: usize,
}

impl<T: TextureType> Kernel<T> {
    pub fn create(width: usize, height: usize, settings: &mut Settings) -> Result<Self, Error> {
        kernel_compilation::refresh_settings(settings)?;
        let mut local_size = 0;
        unsafe {
            gl::GetIntegeri_v(gl::MAX_COMPUTE_WORK_GROUP_SIZE, 0, &mut local_size);
            check_gl()?;
        }
        local_size = local_size.min(64);
        let result = Self {
            kernel: None,
            data: KernelImage::new(width, height)?,
            old_settings: settings.clone(),
            frame: 0,
            local_size: local_size as usize,
        };
        Ok(result)
    }

    pub fn rebuild(&mut self, settings: &mut Settings, force_rebuild: bool) -> Result<(), Error> {
        if settings.check_rebuild() || self.kernel.is_none() || force_rebuild {
            println!("Rebuilding");
            let new_kernel = kernel_compilation::rebuild(settings, self.local_size);
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
        self.data.resize(width, height, self.data.scale)?;
        self.frame = 0;
        Ok(())
    }

    fn update(&mut self, settings: &Settings) -> Result<(), Error> {
        self.data.resize(
            self.data.width,
            self.data.height,
            settings.find("render_scale").unwrap_u32() as usize,
        )?;

        if *settings != self.old_settings {
            self.old_settings = settings.clone();
            self.frame = 0;
        }

        if let Some(kernel) = self.kernel {
            let (width, height) = self.data.size();
            settings.set_uniforms(kernel)?;
            set_arg_u32(kernel, "width", width as u32)?;
            set_arg_u32(kernel, "height", height as u32)?;
            set_arg_u32(kernel, "frame", self.frame)?;
        }

        Ok(())
    }

    fn launch(&mut self) -> Result<(), Error> {
        if let Some(kernel) = self.kernel {
            let (width, height) = self.data.size();
            let total_size = width * height;
            let local_size = self.local_size;
            let launch_size = (total_size + local_size - 1) / local_size;
            unsafe {
                gl::UseProgram(kernel);
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

    pub fn run(&mut self, settings: &mut Settings, force_rebuild: bool) -> Result<(), Error> {
        self.rebuild(settings, force_rebuild)?;
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
