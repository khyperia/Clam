use crate::{
    input::Input,
    kernel::Kernel,
    kernel_compilation::{SourceInfo, MANDELBOX},
    settings::{KeyframeList, Settings},
    Key,
};
use failure::Error;
use khygl::texture::{Texture, TextureType};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub struct SyncInteractiveKernel<T: TextureType> {
    reload: Arc<AtomicBool>,
    source_info: SourceInfo,
    pub kernel: Kernel<T>,
    pub settings: Settings,
    pub keyframes: KeyframeList,
    pub input: Input,
}

impl<T: TextureType> SyncInteractiveKernel<T> {
    pub fn create(width: usize, height: usize) -> Result<Self, Error> {
        let source_info = MANDELBOX;
        let reload = Arc::new(AtomicBool::new(false));
        let reload2 = Arc::downgrade(&reload);
        // TODO: stop watching once `self` dies
        source_info.watch(move || match reload2.upgrade() {
            Some(r) => {
                r.store(true, Ordering::Relaxed);
                true
            }
            None => false,
        });

        let realized_source = source_info.get()?;
        let settings = realized_source.default_settings().clone();
        let keyframes = KeyframeList::load("keyframes.clam5", &realized_source)
            .unwrap_or_else(|_| KeyframeList::new());
        let input = Input::new();
        let kernel = Kernel::create(realized_source, width, height)?;
        let result = Self {
            reload,
            source_info,
            kernel,
            settings,
            keyframes,
            input,
        };
        Ok(result)
    }

    pub fn key_down(&mut self, key: Key) {
        self.input.key_down(
            key,
            &mut self.settings,
            &mut self.keyframes,
            self.kernel.realized_source(),
        );
    }

    pub fn key_up(&mut self, key: Key) {
        self.input.key_up(key, &mut self.settings, &self.keyframes);
    }

    pub fn resize(&mut self, width: usize, height: usize) -> Result<(), Error> {
        self.kernel.resize(width, height)
    }

    pub fn launch(&mut self) -> Result<(), Error> {
        self.input.integrate(&mut self.settings, &self.keyframes);
        if self.reload.swap(false, Ordering::Relaxed) {
            let realized_source = self.source_info.get()?;
            let mut new_settings = realized_source.default_settings().clone();
            new_settings.apply(&self.settings);
            self.settings = new_settings;
            self.kernel.set_src(realized_source);
        }
        self.kernel.run(&self.settings)?;
        Ok(())
    }

    pub fn texture(&mut self) -> &Texture<T> {
        self.kernel.texture()
    }

    pub fn status(&self) -> String {
        self.settings.status(&self.input)
    }
}
