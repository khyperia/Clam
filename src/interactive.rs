use crate::{input::Input, kernel::Kernel, keyframe_list::KeyframeList, settings::Settings, Key};

pub struct SyncInteractiveKernel {
    pub kernel: Kernel,
    pub settings: Settings,
    pub default_settings: Settings,
    pub keyframes: KeyframeList,
    pub input: Input,
}

impl SyncInteractiveKernel {
    pub fn create(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let default_settings = Settings::get_default();
        let keyframes = KeyframeList::load("keyframes.clam5", default_settings.clone())
            .unwrap_or_else(|_| KeyframeList::new());
        let input = Input::new();
        let kernel = Kernel::create(device, queue, width, height);
        Self {
            kernel,
            settings: default_settings.clone(),
            default_settings,
            keyframes,
            input,
        }
    }

    pub fn key_down(&mut self, key: Key) {
        self.input.key_down(
            key,
            &mut self.settings,
            &self.default_settings,
            &mut self.keyframes,
        );
    }

    pub fn key_up(&mut self, key: Key) {
        self.input.key_up(key, &mut self.settings, &self.keyframes);
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.kernel.resize(device, width, height)
    }

    pub fn run(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.input.integrate(&mut self.settings, &self.keyframes);
        self.kernel.run(device, encoder, &self.settings);
    }

    pub fn texture(&self) -> &wgpu::Texture {
        self.kernel.texture()
    }

    pub fn status(&self) -> String {
        self.input.settings_input.status(&self.settings)
    }
}
