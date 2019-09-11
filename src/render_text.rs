use crate::check_gl;
use crate::render_texture::TextureRenderer;
use failure::err_msg;
use failure::Error;
use gl;
use gl::types::*;
use rusttype::{point, FontCollection, PositionedGlyph, Scale};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

const OFFSET: u8 = 33;
const MAX: u8 = 127;

pub struct AtlasEntry {
    pub texture: GLuint,
    image_width: i32,
    image_height: i32,
    x_pos: i32,
    y_pos: i32,
    stride: i32,
}

pub struct TextRenderer {
    spacing: i32,
    pub atlas: Vec<AtlasEntry>,
}

impl TextRenderer {
    pub fn new(rgb: (f32, f32, f32)) -> Result<Self, Error> {
        let font_data = load_font()?;
        let collection = FontCollection::from_bytes(&font_data)?;
        let font = collection.into_font()?;

        let height: f32 = 20.0;

        let scale = Scale {
            x: height,
            y: height,
        };

        // The origin of a line of text is at the baseline (roughly where
        // non-descending letters sit). We don't want to clip the text, so we shift
        // it down with an offset when laying it out. v_metrics.ascent is the
        // distance between the baseline and the highest edge of any glyph in
        // the font. That's enough to guarantee that there's no clipping.
        let v_metrics = font.v_metrics(scale);
        let offset = point(0.0, v_metrics.ascent);
        let spacing = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap;

        let string = (OFFSET..MAX).map(|c| c as char).collect::<String>();
        let atlas = font
            .layout(&string, scale, offset)
            .map(|glyph| render_char(glyph, rgb))
            .collect::<Result<Vec<_>, Error>>()?;

        Ok(Self {
            atlas,
            spacing: spacing as i32,
        })
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
            } else if ch == ' ' {
                x += self.atlas[('*' as u8 - OFFSET) as usize].stride;
            } else {
                let tex = &self.atlas[(ch as u8 - OFFSET) as usize];
                renderer.render(
                    tex.texture,
                    (x + tex.x_pos) as f32 / screen_width as f32,
                    1.0 - (y + tex.y_pos) as f32 / screen_height as f32,
                    tex.image_width as f32 / screen_width as f32,
                    0.0 - tex.image_height as f32 / screen_height as f32,
                )?;
                x += tex.stride;
            }
            max_x = max_x.max(x);
        }
        Ok(())
    }
}

fn render_char(glyph: PositionedGlyph, rgb: (f32, f32, f32)) -> Result<AtlasEntry, Error> {
    //glyph.set_position(Point { x: 0.0, y: 0.0 });
    let bb = glyph
        .pixel_bounding_box()
        .expect("Could not get bounding box of glyph");
    let h_metrics = glyph.unpositioned().h_metrics();
    let width = bb.width();
    let height = bb.height();

    let mut pixels = vec![0f32; width as usize * height as usize * (4 * 4)];

    glyph.draw(|x, y, v| {
        let index = (y as usize * width as usize + x as usize) * 4;
        pixels[index] = rgb.0;
        pixels[index + 1] = rgb.1;
        pixels[index + 2] = rgb.2;
        pixels[index + 3] = v;
    });

    let mut texture = 0;
    unsafe {
        gl::CreateTextures(gl::TEXTURE_2D, 1, &mut texture);
        check_gl()?;
        gl::TextureStorage2D(texture, 1, gl::RGBA32F, width as _, height as _);
        check_gl()?;
    }

    check_gl()?;
    unsafe {
        gl::TextureSubImage2D(
            texture,
            0,
            0,
            0,
            width as i32,
            height as i32,
            gl::RGBA,
            gl::FLOAT,
            pixels.as_ptr() as *const _,
        );
    }
    check_gl()?;

    Ok(AtlasEntry {
        texture,
        image_width: width,
        image_height: height,
        x_pos: h_metrics.left_side_bearing.ceil() as i32,
        y_pos: bb.min.y,
        stride: h_metrics.advance_width.ceil() as i32,
    })
}

fn load_font() -> Result<Vec<u8>, Error> {
    let path = find_font()?;
    let mut file = File::open(path)?;
    let mut contents = vec![];
    file.read_to_end(&mut contents)?;
    Ok(contents)
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
