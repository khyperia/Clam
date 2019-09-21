use crate::check_gl;
use failure::Error;
use gl::types::*;
use std::{
    ffi::{CStr, CString},
    marker::PhantomData,
    ptr::null_mut,
};

pub trait TextureType: Clone + Default {
    fn internalformat() -> GLuint;
    fn format() -> GLuint;
    fn type_() -> GLuint;
    fn size() -> usize {
        std::mem::size_of::<Self>()
    }
}

impl TextureType for [f32; 4] {
    fn internalformat() -> GLuint {
        gl::RGBA32F
    }
    fn format() -> GLuint {
        gl::RGBA
    }
    fn type_() -> GLuint {
        gl::FLOAT
    }
}

impl TextureType for [u8; 4] {
    fn internalformat() -> GLuint {
        gl::RGBA8UI
    }
    fn format() -> GLuint {
        gl::RGBA
    }
    fn type_() -> GLuint {
        gl::UNSIGNED_INT_8_8_8_8
    }
}

impl TextureType for [u32; 2] {
    fn internalformat() -> GLuint {
        gl::RG32UI
    }
    fn format() -> GLuint {
        gl::RG_INTEGER
    }
    fn type_() -> GLuint {
        gl::UNSIGNED_INT
    }
}

impl TextureType for u32 {
    fn internalformat() -> GLuint {
        gl::R32UI
    }
    fn format() -> GLuint {
        gl::RED_INTEGER
    }
    fn type_() -> GLuint {
        gl::UNSIGNED_INT
    }
}

pub struct Texture<T: TextureType> {
    pub id: GLuint,
    pub size: (usize, usize),
    _t: PhantomData<T>,
}

impl<T: TextureType> Texture<T> {
    pub fn new(width: usize, height: usize) -> Result<Self, Error> {
        let format = T::internalformat();
        let mut texture = 0;
        unsafe {
            gl::CreateTextures(gl::TEXTURE_2D, 1, &mut texture);
            check_gl()?;
            gl::TextureStorage2D(texture, 1, format, width as _, height as _);
            check_gl()?;
            gl::TextureParameteri(texture, gl::TEXTURE_MIN_FILTER, gl::NEAREST as GLint);
            check_gl()?;
            gl::TextureParameteri(texture, gl::TEXTURE_MAG_FILTER, gl::NEAREST as GLint);
            check_gl()?;
        }
        Ok(Self {
            id: texture,
            size: (width, height),
            _t: PhantomData,
        })
    }

    pub fn download(&mut self) -> Result<CpuTexture<T>, Error> {
        let mut pixels = vec![T::default(); self.size.0 * self.size.1];
        let buf_size = T::size() * pixels.len();
        unsafe {
            gl::GetTextureImage(
                self.id,
                0,
                T::format(),
                T::type_(),
                buf_size as i32,
                pixels.as_mut_ptr() as *mut _,
            );
            check_gl()?;
        }
        Ok(CpuTexture::new(pixels, self.size.0, self.size.1))
    }

    pub fn bind(&self, unit: usize) -> Result<(), Error> {
        unsafe {
            gl::BindImageTexture(
                unit as GLuint,
                self.id,
                0,
                gl::FALSE,
                0,
                gl::READ_WRITE,
                T::internalformat(),
            );
            check_gl()
        }
    }
}

impl<T: TextureType> Drop for Texture<T> {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.id);
        }
        check_gl().expect("Failed to delete texture in drop impl");
    }
}

pub struct CpuTexture<T> {
    pub data: Vec<T>,
    pub width: usize,
    pub height: usize,
}

impl<T> CpuTexture<T> {
    pub fn new(data: Vec<T>, width: usize, height: usize) -> Self {
        Self {
            data,
            width,
            height,
        }
    }
}

fn get_uniform_location(kernel: GLuint, key: &str) -> GLint {
    let key = CString::new(key).expect("Failed to convert uniform name to null-terminated string");
    unsafe { gl::GetUniformLocation(kernel, key.as_ptr() as *const GLchar) }
}

pub fn set_arg_f32(kernel: GLuint, key: &str, value: f32) -> Result<(), Error> {
    let location = get_uniform_location(kernel, key);
    if location != -1 {
        unsafe {
            gl::UseProgram(kernel);
            gl::Uniform1f(location, value);
            gl::UseProgram(0);
        }
    }
    check_gl()?;
    Ok(())
}

pub fn set_arg_u32(kernel: GLuint, key: &str, value: u32) -> Result<(), Error> {
    let location = get_uniform_location(kernel, key);
    if location != -1 {
        unsafe {
            gl::UseProgram(kernel);
            gl::Uniform1ui(location, value);
            gl::UseProgram(0);
        }
    }
    check_gl()?;
    Ok(())
}

pub unsafe fn create_compute_program(sources: &[&str]) -> Result<GLuint, Error> {
    let shader = create_shader(sources, gl::COMPUTE_SHADER)?;
    create_program(&[shader])
}

pub unsafe fn create_vert_frag_program(
    vertex: &[&str],
    fragment: &[&str],
) -> Result<GLuint, Error> {
    let vertex = create_shader(vertex, gl::VERTEX_SHADER)?;
    let fragment = create_shader(fragment, gl::FRAGMENT_SHADER)?;
    create_program(&[vertex, fragment])
}

pub unsafe fn create_program(shaders: &[GLuint]) -> Result<GLuint, Error> {
    let program = gl::CreateProgram();
    for &shader in shaders {
        gl::AttachShader(program, shader);
    }
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

    for &shader in shaders {
        gl::DeleteShader(shader);
    }

    check_gl()?;

    Ok(program)
}

pub unsafe fn create_shader(sources: &[&str], shader_type: GLenum) -> Result<GLuint, Error> {
    let shader = gl::CreateShader(shader_type);
    check_gl()?;
    let vec_sources = sources
        .iter()
        .map(|source| source.as_ptr() as *const GLchar)
        .collect::<Vec<_>>();
    let lengths = sources
        .iter()
        .map(|source| source.len() as GLint)
        .collect::<Vec<_>>();
    gl::ShaderSource(
        shader,
        vec_sources.len() as GLsizei,
        vec_sources.as_ptr() as *const *const GLchar,
        lengths.as_ptr(),
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
