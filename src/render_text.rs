use crate::check_gl;
use crate::render_texture::TextureRenderer;
use failure::err_msg;
use failure::Error;
use gl;
use gl::types::*;
use sdl2::pixels::Color;
use sdl2::pixels::PixelFormat;
use sdl2::ttf;
use sdl2::ttf::Font;
use std::path::Path;

const OFFSET: u8 = 32;

pub struct AtlasEntry {
    pub texture: GLuint,
    width: u32,
    height: u32,
}

pub struct TextRenderer {
    spacing: i32,
    pub atlas: Vec<AtlasEntry>,
}

impl TextRenderer {
    pub fn new(color: Color) -> Self {
        let ttf = ttf::init().expect("Couldn't init sdl2_ttf");
        let font = ttf
            .load_font(find_font().expect("Couldn't find font"), 20)
            .expect("Couldn't load font");

        let spacing = font.recommended_line_spacing();
        let format = unsafe {
            PixelFormat::from_ll(sdl2::sys::SDL_AllocFormat(
                sdl2::sys::SDL_PIXELFORMAT_RGBA8888 as u32,
            ))
        };
        let atlas = (OFFSET..255)
            .map(|x| render_char(&font, &format, color, x as char))
            .collect::<Vec<_>>();
        unsafe { sdl2::sys::SDL_FreeFormat(format.raw()) };
        Self { spacing, atlas }
    }

    pub fn render(
        &self,
        renderer: &TextureRenderer,
        text: &str,
        screen_width: u32,
        screen_height: u32,
    ) -> Result<(), Error> {
        let mut line_x = 10;
        let mut x = line_x;
        let mut max_x = x;
        let mut y = 10;
        for ch in text.chars() {
            if ch == '\n' {
                y += self.spacing;
                if y + self.spacing > screen_height as i32 {
                    y = 10;
                    line_x += max_x;
                }
                x = line_x;
            } else {
                let tex = &self.atlas[(ch as u8 - OFFSET) as usize];
                renderer.render(
                    tex.texture,
                    x as f32 / screen_width as f32,
                    1.0 - y as f32 / screen_height as f32,
                    tex.width as f32 / screen_width as f32,
                    0.0 - tex.height as f32 / screen_height as f32,
                )?;
                x += tex.width;
            }
            max_x = max_x.max(x);
        }
        Ok(())
    }
}

fn render_char(font: &Font, format: &PixelFormat, color: Color, ch: char) -> AtlasEntry {
    lazy_static! {};
    let rendered = font
        .render_char(ch)
        .blended(color)
        .expect("Couldn't render char")
        .convert(format)
        .expect("Couldn't convert format");
    let width = rendered.width();
    let height = rendered.height();
    let mut texture = 0;
    unsafe {
        gl::CreateTextures(gl::TEXTURE_2D, 1, &mut texture);
        check_gl().unwrap();
        gl::TextureStorage2D(texture, 1, gl::RGBA32F, width as _, height as _);
        check_gl().unwrap();
    }

    rendered.with_lock(|pixels| {
        check_gl().unwrap();
        unsafe {
            gl::TextureSubImage2D(
                texture,
                0,
                0,
                0,
                width as i32,
                height as i32,
                gl::RGBA,
                gl::UNSIGNED_INT_8_8_8_8,
                pixels.as_ptr() as *const _,
            );
        }
        check_gl().unwrap();
    });
    AtlasEntry {
        texture,
        width,
        height,
    }
}

fn find_font() -> Result<&'static Path, Error> {
    let locations: [&'static Path; 6] = [
        "/usr/share/fonts/TTF/FiraMono-Regular.ttf".as_ref(),
        "/usr/share/fonts/TTF/FiraSans-Regular.ttf".as_ref(),
        "C:\\Windows\\Fonts\\arial.ttf".as_ref(),
        "/usr/share/fonts/TTF/DejaVuSans.ttf".as_ref(),
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf".as_ref(),
        "/Library/Fonts/Andale Mono.ttf".as_ref(),
    ];
    for &location in &locations {
        if location.exists() {
            return Ok(location);
        }
    }
    Err(err_msg("No font found"))
}
