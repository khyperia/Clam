use crate::{check_gl, gl_help};
use failure::Error;
use gl::{self, types::*};

// https://rauwendaal.net/2014/06/14/rendering-a-screen-covering-triangle-in-opengl/

#[derive(Clone, Copy)]
pub enum TextureRendererKind {
    #[cfg(feature = "vr")]
    U8,
    F32,
}

pub struct TextureRenderer {
    program: GLuint,
    pos_size_location: GLint,
}

impl TextureRenderer {
    pub fn new(kind: TextureRendererKind) -> Result<Self, Error> {
        check_gl()?;
        let frag = match kind {
            #[cfg(feature = "vr")]
            TextureRendererKind::U8 => FRAGMENT_SHADER_U8,
            TextureRendererKind::F32 => FRAGMENT_SHADER_F32,
        };
        let program = unsafe { gl_help::create_vert_frag_program(&[VERTEX_SHADER], &[frag])? };
        let pos_size_location =
            unsafe { gl::GetUniformLocation(program, b"pos_size\0".as_ptr() as *const GLchar) };
        check_gl()?;
        if pos_size_location == -1 {
            panic!("pos_size_location not found");
        }
        unsafe {
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::BLEND);
            gl::Enable(gl::TEXTURE_2D);
        }
        check_gl()?;
        Ok(Self {
            program,
            pos_size_location,
        })
    }

    pub fn render(
        &self,
        texture: GLuint,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> Result<(), Error> {
        unsafe {
            gl::UseProgram(self.program);
            gl::Uniform4f(self.pos_size_location, x, y, width, height);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);
            gl::BindTexture(gl::TEXTURE_2D, 0);
            gl::UseProgram(0);
            check_gl()?;
        }
        Ok(())
    }
}

impl Drop for TextureRenderer {
    fn drop(&mut self) {
        unsafe { gl::DeleteProgram(self.program) }
    }
}

const VERTEX_SHADER: &str = "
#version 130

uniform vec4 pos_size;
out vec2 texCoord;

void main()
{
    float x = (gl_VertexID & 1);
    float y = (gl_VertexID & 2) >> 1;
    texCoord.x = x;
    texCoord.y = y;
    x = pos_size.x + pos_size.z * x;
    y = pos_size.y + pos_size.w * y;
    gl_Position = vec4(x*2-1, y*2-1, 0, 1);
}
";

const FRAGMENT_SHADER_F32: &str = "
#version 130

uniform sampler2D tex;
in vec2 texCoord;

void main()
{
    vec4 color1 = texture(tex, texCoord);
    gl_FragColor = color1;
}
";

#[cfg(feature = "vr")]
const FRAGMENT_SHADER_U8: &str = "
#version 130

uniform usampler2D tex;
in vec2 texCoord;

void main()
{
    vec4 color1 = texture(tex, texCoord);
    gl_FragColor = color1 / 255.0;
}
";
