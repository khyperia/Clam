use crate::check_gl;
use failure::Error;
use gl;
use gl::types::*;
use std::ffi::CStr;
use std::ptr::null;
use std::ptr::null_mut;

// https://rauwendaal.net/2014/06/14/rendering-a-screen-covering-triangle-in-opengl/

pub struct TextureRenderer {
    program: GLuint,
    pos_size_location: GLint,
}

impl TextureRenderer {
    pub fn new() -> Self {
        check_gl().unwrap();
        let program = unsafe { create_program() };
        let pos_size_location =
            unsafe { gl::GetUniformLocation(program, b"pos_size\0".as_ptr() as *const i8) };
        check_gl().unwrap();
        if pos_size_location == -1 {
            panic!("pos_size_location not found");
        }
        unsafe {
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::BLEND);
        }
        check_gl().unwrap();
        Self {
            program,
            pos_size_location,
        }
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
        }
        Ok(())
    }
}

impl Drop for TextureRenderer {
    fn drop(&mut self) {
        unsafe { gl::DeleteProgram(self.program) }
    }
}

unsafe fn create_program() -> GLuint {
    let vertex = create_shader(VERTEX_SHADER, gl::VERTEX_SHADER);
    let fragment = create_shader(FRAGMENT_SHADER, gl::FRAGMENT_SHADER);

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
    check_gl().unwrap();

    // ???
    gl::DeleteShader(vertex);
    gl::DeleteShader(fragment);

    check_gl().unwrap();

    program
}

unsafe fn create_shader(source: &'static [u8], shader_type: GLenum) -> GLuint {
    let shader = gl::CreateShader(shader_type);
    check_gl().unwrap();
    gl::ShaderSource(
        shader,
        1,
        &(source.as_ptr()) as *const *const u8 as *const *const i8,
        null(),
    );
    check_gl().unwrap();
    gl::CompileShader(shader);
    check_gl().unwrap();
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
    check_gl().unwrap();
    shader
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

const FRAGMENT_SHADER: &[u8] = b"
uniform sampler2D tex;
in vec2 texCoord;

void main()
{
    vec4 color1 = texture2D(tex, texCoord);
    // if (color1.w == 0) {
    //     color1 = vec4(1, 0, 1, 1);
    // }
    gl_FragColor = color1;
}
\0";
