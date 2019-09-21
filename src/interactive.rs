use crate::{
    gl_help::{Texture, TextureType},
    input::Input,
    kernel::FractalKernel,
    kernel_compilation,
    settings::Settings,
    Key,
};
use failure::Error;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub struct SyncInteractiveKernel<T: TextureType> {
    rebuild: Arc<AtomicBool>,
    pub kernel: FractalKernel<T>,
    pub settings: Settings,
    pub input: Input,
}

impl<T: TextureType> SyncInteractiveKernel<T> {
    pub fn create(width: usize, height: usize) -> Result<Self, Error> {
        let rebuild = Arc::new(AtomicBool::new(false));
        let rebuild2 = rebuild.clone();
        // TODO: stop watching once `self` dies
        kernel_compilation::watch_src(move || rebuild2.store(true, Ordering::Relaxed));

        let mut settings = Settings::new();
        let input = Input::new();
        let kernel = FractalKernel::create(width, height, &mut settings)?;
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

    pub fn resize(&mut self, width: usize, height: usize) -> Result<(), Error> {
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

    pub fn texture(&mut self) -> Result<&Texture<T>, Error> {
        let image = self.kernel.texture()?;
        Ok(image)
    }

    // pub fn download(&mut self) -> Result<CpuTexture<T>, Error> {
    //     let image = self.kernel.download()?;
    //     Ok(image)
    // }

    pub fn status(&self) -> String {
        self.settings.status(&self.input)
    }
}
