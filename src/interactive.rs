use crate::input::Input;
use crate::kernel::FractalKernel;
use crate::kernel_compilation;
use crate::settings::Settings;
use failure::Error;
use gl::types::GLuint;
use ocl::OclPrm;
use sdl2::keyboard::Scancode as Key;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub struct ImageData<T: OclPrm> {
    pub data_cpu: Option<Vec<T>>,
    pub data_gl: Option<GLuint>,
    pub width: u32,
    pub height: u32,
}

impl<T: OclPrm> ImageData<T> {
    pub fn new(data_cpu: Option<Vec<T>>, data_gl: Option<GLuint>, width: u32, height: u32) -> Self {
        Self {
            data_cpu,
            data_gl,
            width,
            height,
        }
    }
}

pub struct SyncInteractiveKernel<T: OclPrm> {
    rebuild: Arc<AtomicBool>,
    pub kernel: FractalKernel<T>,
    pub settings: Settings,
    pub input: Input,
}

impl<T: OclPrm> SyncInteractiveKernel<T> {
    pub fn create(width: u32, height: u32, is_ogl: bool) -> Result<Self, Error> {
        let rebuild = Arc::new(AtomicBool::new(false));
        let rebuild2 = rebuild.clone();
        // TODO: stop watching once `self` dies
        kernel_compilation::watch_src(move || rebuild2.store(true, Ordering::Relaxed));

        let mut settings = Settings::new();
        let input = Input::new();
        let kernel = FractalKernel::create(width, height, is_ogl, &mut settings).unwrap();
        let result = Self {
            kernel,
            rebuild,
            settings,
            input,
        };
        Ok(result)
    }

    pub fn key_down(&mut self, key: Key) {
        self.input.key_down(key, &mut self.settings);
    }

    pub fn key_up(&mut self, key: Key) {
        self.input.key_up(key, &mut self.settings);
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        self.kernel.resize(width, height)
    }

    pub fn launch(&mut self) -> Result<(), Error> {
        self.input.integrate(&mut self.settings);
        self.kernel.run(
            &mut self.settings,
            self.rebuild.swap(false, Ordering::Relaxed),
        )?;
        Ok(())
    }

    pub fn download(&mut self) -> Result<ImageData<T>, Error> {
        let image = self.kernel.download()?;
        Ok(image)
    }

    pub fn status(&self) -> String {
        self.settings.status(&self.input)
    }
}
