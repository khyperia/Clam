use crate::{
    check_gl,
    display::{run_display, Display},
    fps_counter::FpsCounter,
    interactive::SyncInteractiveKernel,
    render_text::TextRenderer,
    render_texture::{TextureRenderer, TextureRendererKind},
    Key,
};
use failure::Error;

struct GlDisplay {
    interactive_kernel: SyncInteractiveKernel<[f32; 4]>,
    texture_renderer: TextureRenderer,
    text_renderer: TextRenderer,
    fps: FpsCounter,
    width: usize,
    height: usize,
}

impl Display for GlDisplay {
    fn setup(width: usize, height: usize) -> Result<Self, Error> {
        let interactive_kernel = SyncInteractiveKernel::<[f32; 4]>::create(width, height)?;

        let texture_renderer = TextureRenderer::new(TextureRendererKind::F32)?;
        let text_renderer = TextRenderer::new((1.0, 0.75, 0.75))?;

        let fps = FpsCounter::new(1.0);

        Ok(Self {
            interactive_kernel,
            texture_renderer,
            text_renderer,
            fps,
            width,
            height,
        })
    }

    fn render(&mut self) -> Result<(), Error> {
        self.interactive_kernel.launch()?;
        let img = self.interactive_kernel.texture()?;

        self.texture_renderer.render(img.id, 0.0, 0.0, 1.0, 1.0)?;

        let display = format!(
            "{:.2} fps\n{}",
            self.fps.value(),
            self.interactive_kernel.status()
        );
        self.text_renderer
            .render(&self.texture_renderer, &display, self.width, self.height)?;
        self.fps.tick();
        check_gl()?;
        Ok(())
    }

    fn resize(&mut self, width: usize, height: usize) -> Result<(), Error> {
        self.width = width;
        self.height = height;
        self.interactive_kernel.resize(width, height)?;
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
}

pub fn gl_display(width: f64, height: f64) -> Result<(), Error> {
    run_display::<GlDisplay>(width, height)
}
