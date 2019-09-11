use crate::check_gl;
use crate::display;
use crate::display::Display;
use crate::fps_counter::FpsCounter;
use crate::interactive::SyncInteractiveKernel;
use crate::render_text::TextRenderer;
use crate::render_texture::TextureRenderer;
use crate::render_texture::TextureRendererKind;
use crate::Key;
use failure::Error;

struct GlDisplay {
    interactive_kernel: SyncInteractiveKernel<f32>,
    texture_renderer: TextureRenderer,
    text_renderer: TextRenderer,
    fps: FpsCounter,
    width: u32,
    height: u32,
}

impl Display for GlDisplay {
    fn setup(width: u32, height: u32) -> Result<Self, Error> {
        let interactive_kernel = SyncInteractiveKernel::<f32>::create(width, height, true)?;

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
        let img = self.interactive_kernel.download()?;

        self.texture_renderer.render(
            img.data_gl.expect("gl_display needs OGL texture"),
            0.0,
            0.0,
            1.0,
            1.0,
        )?;

        let display = format!(
            "{:.2} fps\n{}",
            self.fps.value(),
            self.interactive_kernel.status()
        );
        self.text_renderer
            .render(&self.texture_renderer, &display, self.width, self.height)?;
        //gl.draw_frame([1.0, 0.5, 0.7, 1.0]);
        self.fps.tick();
        check_gl()?;
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        self.width = width as u32;
        self.height = height as u32;
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
    display::run_display::<GlDisplay>(width, height)
}
