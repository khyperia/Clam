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
    output: Option<Texture<T>>,
    scratch: Option<Texture<[f32; 4]>>,
    randbuf: Option<Texture<u32>>,
}

impl<T: TextureType> KernelImage<T> {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            scale: 1,
            output: None,
            scratch: None,
            randbuf: None,
        }
    }

    fn size(&self) -> (usize, usize) {
        (self.width / self.scale, self.height / self.scale)
    }

    fn data(&mut self) -> Result<(&Texture<T>, &Texture<[f32; 4]>, &Texture<u32>), Error> {
        let (width, height) = self.size();
        if self.output.is_none() {
            self.output = Some(Texture::new(width, height)?);
        }
        if self.scratch.is_none() {
            self.scratch = Some(Texture::new(width, height)?);
        }
        if self.randbuf.is_none() {
            self.randbuf = Some(Texture::new(width, height)?);
        }
        Ok((
            self.output.as_ref().expect("Didn't assign output?"),
            self.scratch.as_ref().expect("Didn't assign scratch?"),
            self.randbuf.as_ref().expect("Didn't assign randbuf?"),
        ))
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
        if old_size != self.size() {
            self.output = None;
            self.scratch = None;
            self.randbuf = None;
        }
        Ok(())
    }

    fn download(&mut self) -> Result<CpuTexture<T>, Error> {
        self.output
            .as_mut()
            .ok_or_else(|| failure::err_msg("Cannot download image that hasn't been created yet"))
            .and_then(|img| img.download())
    }
}

pub struct FractalKernel<T: TextureType> {
    kernel: Option<GLuint>,
    data: KernelImage<T>,
    old_settings: Settings,
    frame: u32,
    local_size: usize,
}

impl<T: TextureType> FractalKernel<T> {
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
            data: KernelImage::new(width, height),
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
            let (texture, scratch, randbuf) = self.data.data()?;
            let total_size = width * height;
            let local_size = self.local_size;
            let launch_size = (total_size + local_size - 1) / local_size;
            unsafe {
                gl::UseProgram(kernel);
                check_gl()?;
                texture.bind(0)?;
                scratch.bind(1)?;
                randbuf.bind(2)?;
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

    pub fn texture(&mut self) -> Result<&Texture<T>, Error> {
        Ok(self.data.data()?.0)
    }

    pub fn download(&mut self) -> Result<CpuTexture<T>, Error> {
        self.data.download()
    }

    pub fn sync_renderer(&mut self) -> Result<(), Error> {
        unsafe {
            gl::Finish();
            check_gl()
        }
    }
}
