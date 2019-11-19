use crate::{
    check_gl, display, display::Display, fps_counter::FpsCounter,
    interactive::SyncInteractiveKernel, Key,
};
use failure::Error;
use khygl::{render_text::TextRenderer, render_texture::TextureRenderer};

struct GlDisplay {
    interactive_kernel: SyncInteractiveKernel<[f32; 4]>,
    texture_renderer: TextureRenderer,
    text_renderer: TextRenderer,
    fps: FpsCounter,
    width: usize,
    height: usize,
}

impl Display for GlDisplay {
    fn setup(size: (usize, usize), _: f64) -> Result<Self, Error> {
        let interactive_kernel = SyncInteractiveKernel::<[f32; 4]>::create(size.0, size.1)?;

        let texture_renderer = TextureRenderer::new()?;
        let text_renderer = TextRenderer::new(20.0)?;

        let fps = FpsCounter::new(1.0);

        Ok(Self {
            interactive_kernel,
            texture_renderer,
            text_renderer,
            fps,
            width: size.0,
            height: size.1,
        })
    }

    fn render(&mut self) -> Result<(), Error> {
        self.interactive_kernel.launch()?;
        let img = self.interactive_kernel.texture();

        self.texture_renderer
            .render(&img, (self.width as f32, self.height as f32))
            .go()?;

        let display = format!(
            "{:.2} fps\n{}",
            self.fps.value(),
            self.interactive_kernel.status()
        );
        self.text_renderer.render(
            &self.texture_renderer,
            &display,
            [1.0, 0.75, 0.75, 1.0],
            (10, 10),
            (self.width, self.height),
        )?;
        self.fps.tick();
        check_gl()?;
        Ok(())
    }

    fn resize(&mut self, size: (usize, usize)) -> Result<(), Error> {
        self.width = size.0;
        self.height = size.1;
        self.interactive_kernel.resize(size.0, size.1)?;
        Ok(())
    }

    fn key_up(&mut self, key: Key) -> Result<(), Error> {
        self.interactive_kernel.key_up(key);
        Ok(())
    }

    fn key_down(&mut self, key: Key) -> Result<(), Error> {
        self.interactive_kernel.key_down(key);
        Ok(())
    }

    fn received_character(&mut self, _: char) -> Result<(), Error> {
        Ok(())
    }
}

pub fn run(width: f64, height: f64) -> Result<(), Error> {
    display::run::<GlDisplay>((width, height))
}
