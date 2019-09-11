use crate::check_gl;
use failure::Error;
use gl;
use gl::types::*;
use std::ffi::CStr;
use std::ptr::null;
use std::ptr::null_mut;

// https://rauwendaal.net/2014/06/14/rendering-a-screen-covering-triangle-in-opengl/

pub enum TextureRendererKind {
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
            TextureRendererKind::U8 => FRAGMENT_SHADER_U8,
            TextureRendererKind::F32 => FRAGMENT_SHADER_F32,
        };
        let program = unsafe { create_program(VERTEX_SHADER, frag)? };
        let pos_size_location =
            unsafe { gl::GetUniformLocation(program, b"pos_size\0".as_ptr() as *const i8) };
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
            check_gl()?;
            gl::UseProgram(self.program);
            check_gl()?;
            gl::Uniform4f(self.pos_size_location, x, y, width, height);
            check_gl()?;
            gl::BindTexture(gl::TEXTURE_2D, texture);
            check_gl()?;
            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);
            check_gl()?;
            gl::BindTexture(gl::TEXTURE_2D, 0);
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

unsafe fn create_program(vertex: &'static [u8], fragment: &'static [u8]) -> Result<GLuint, Error> {
    let vertex = create_shader(vertex, gl::VERTEX_SHADER)?;
    let fragment = create_shader(fragment, gl::FRAGMENT_SHADER)?;

    let program = gl::CreateProgram();
    gl::AttachShader(program, vertex);
    gl::AttachShader(program, fragment);
    gl::LinkProgram(program);

    let mut success = 0;
    gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
    if success == 0 {
        let mut info_log: [GLchar; 512] = [0; 512];
        let ptr = info_log.as_mut_ptr();
        gl::GetProgramInfoLog(program, 512, null_mut(), ptr);
        let log = CStr::from_ptr(ptr)
            .to_str()
            .expect("Invalid OpenGL error message");
        panic!("Failed to compile OpenGL program:\n{}", log);
    }
    check_gl()?;

    // ???
    gl::DeleteShader(vertex);
    gl::DeleteShader(fragment);

    check_gl()?;

    Ok(program)
}

unsafe fn create_shader(source: &'static [u8], shader_type: GLenum) -> Result<GLuint, Error> {
    let shader = gl::CreateShader(shader_type);
    check_gl()?;
    gl::ShaderSource(
        shader,
        1,
        &(source.as_ptr()) as *const *const u8 as *const *const i8,
        null(),
    );
    check_gl()?;
    gl::CompileShader(shader);
    check_gl()?;
    let mut success = 0;
    gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
    if success == 0 {
        let mut info_log: [GLchar; 512] = [0; 512];
        let ptr = info_log.as_mut_ptr();
        gl::GetShaderInfoLog(shader, 512, null_mut(), ptr);
        let log = CStr::from_ptr(ptr)
            .to_str()
            .expect("Invalid OpenGL error message");
        panic!("Failed to compile OpenGL shader:\n{}", log);
    }
    check_gl()?;
    Ok(shader)
}

const VERTEX_SHADER: &[u8] = b"
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
\0";

const FRAGMENT_SHADER_F32: &[u8] = b"
uniform sampler2D tex;
in vec2 texCoord;

void main()
{
    vec4 color1 = texture(tex, texCoord);
    gl_FragColor = color1;
}
\0";

const FRAGMENT_SHADER_U8: &[u8] = b"
uniform usampler2D tex;
in vec2 texCoord;

void main()
{
    vec4 color1 = texture(tex, texCoord);
    gl_FragColor = color1 / 255.0;
}
\0";
