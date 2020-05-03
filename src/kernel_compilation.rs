use crate::{
    check_gl, parse_vector3,
    setting_value::{SettingValue, SettingValueEnum},
    settings::Settings,
    Error,
};
use gl::types::*;
use khygl::create_compute_program;
use lazy_static::lazy_static;
use regex::Regex;
use std::{
    ffi::{c_void, CStr},
    fs,
    fs::File,
    io::prelude::*,
    path::Path,
    thread,
    time::{Duration, SystemTime},
};

pub struct SourceInfo {
    source: &'static str,
    path: &'static str,
}

pub const MANDELBOX: SourceInfo = SourceInfo {
    source: include_str!("mandelbox.comp.glsl"),
    path: "src/mandelbox.comp.glsl",
};

impl SourceInfo {
    // the "notify" crate is broken af, so roll our own
    // the function returns false when the thread should die
    pub fn watch<F: Fn() -> bool + Send + 'static>(&self, on_changed: F) {
        let path = self.path;
        thread::spawn(move || {
            fn get_time(path: &str) -> Option<SystemTime> {
                let meta = match fs::metadata(path) {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                let time = match meta.modified() {
                    Ok(v) => v,
                    Err(_) => return None,
                };
                Some(time)
            }
            let mut time = get_time(path);
            loop {
                thread::sleep(Duration::from_secs(1));
                let new_time = get_time(path);
                if new_time != time {
                    time = new_time;
                    if !on_changed() {
                        break;
                    }
                }
            }
        });
    }

    pub fn get(&self) -> Result<RealizedSource, Error> {
        let mut file = match File::open(self.path) {
            Ok(f) => f,
            Err(err) => {
                println!("Warning for {}: {}", self.path, err);
                return Ok(RealizedSource::new(self.source.to_string()));
            }
        };
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(RealizedSource::new(contents))
    }
}
pub struct RealizedSource {
    source: String,
    settings: Settings,
}

impl RealizedSource {
    fn new(source: String) -> Self {
        let settings = settings_from_str(&source);
        Self { source, settings }
    }

    pub fn default_settings(&self) -> &Settings {
        &self.settings
    }

    pub fn rebuild(&self, settings: &Settings, local_size: usize) -> Result<ComputeShader, Error> {
        let mut to_compile = self.source.clone();
        generate_src(&mut to_compile, settings, local_size);
        ComputeShader::new(&[&to_compile])
    }
}

pub struct ComputeShader {
    pub shader: GLuint,
    uniforms: Vec<Uniform>,
}

impl ComputeShader {
    fn new(sources: &[&str]) -> Result<Self, Error> {
        let shader = create_compute_program(sources)?;
        if !shader.success {
            return Err(format!("Failed to compile shader: {}", shader.log).into());
        }
        if !shader.log.is_empty() {
            println!("Shader compilation log: {}", shader.log);
        }
        let shader = shader.shader;
        if let Ok(path) = std::env::var("CLAM5_BINARY") {
            dump_binary(shader, path)?;
        }
        Ok(Self {
            shader,
            uniforms: uniforms(shader)?,
        })
    }

    pub fn uniforms(&self) -> &[Uniform] {
        &self.uniforms
    }
}

impl Drop for ComputeShader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.shader);
        }
    }
}

pub struct Uniform {
    pub name: String,
    pub ty: GLenum,
    pub location: GLint,
    // TODO: unsafe lifetime
    pub program: GLuint,
}

impl Uniform {
    fn new(name: String, ty: GLenum, location: GLint, program: GLuint) -> Self {
        Self {
            name,
            ty,
            location,
            program,
        }
    }

    pub fn set_arg_f32(&self, value: f32) -> Result<(), Error> {
        unsafe {
            gl::UseProgram(self.program);
            gl::Uniform1f(self.location, value);
            gl::UseProgram(0);
        }
        check_gl()?;
        Ok(())
    }

    pub fn set_arg_f32_3(&self, x: f32, y: f32, z: f32) -> Result<(), Error> {
        unsafe {
            gl::UseProgram(self.program);
            gl::Uniform3f(self.location, x, y, z);
            gl::UseProgram(0);
        }
        check_gl()?;
        Ok(())
    }

    pub fn set_arg_u32(&self, value: u32) -> Result<(), Error> {
        unsafe {
            gl::UseProgram(self.program);
            gl::Uniform1ui(self.location, value);
            gl::UseProgram(0);
        }
        check_gl()?;
        Ok(())
    }
}

fn uniforms(program: GLuint) -> Result<Vec<Uniform>, Error> {
    unsafe {
        let mut uniform_length = 0;
        gl::GetProgramiv(program, gl::ACTIVE_UNIFORM_MAX_LENGTH, &mut uniform_length);
        check_gl()?;
        let mut count = 0;
        gl::GetProgramiv(program, gl::ACTIVE_UNIFORMS, &mut count);
        check_gl()?;
        let mut buffer = vec![0; uniform_length as usize];
        let mut results = Vec::new();
        for i in 0..count {
            let mut length = 0;
            let mut size = 0;
            let mut ty = 0;
            gl::GetActiveUniform(
                program,
                i as _,
                uniform_length,
                &mut length,
                &mut size,
                &mut ty,
                buffer.as_mut_ptr(),
            );
            check_gl()?;
            let name_slice = &buffer[0..(length as usize)];
            let name_slice_u8 = &*(name_slice as *const [i8] as *const [u8]);
            let name = std::str::from_utf8(name_slice_u8)?.to_string();
            let location = gl::GetUniformLocation(program, name_slice.as_ptr());
            check_gl()?;
            if location < 0 {
                return Err(format!("Could not find uniform: {}", name).into());
            }
            results.push(Uniform::new(name, ty, location, program));
        }
        Ok(results)
    }
}

fn shader_version() -> Result<String, Error> {
    let ver = unsafe { CStr::from_ptr(gl::GetString(gl::SHADING_LANGUAGE_VERSION) as _) };
    check_gl()?;
    let ver = ver
        .to_str()?
        .split_whitespace()
        .next()
        .ok_or_else(|| "GL_SHADING_LANGUAGE_VERSION is empty")?
        .replace(".", "");
    Ok(ver)
}

fn generate_src(original_source: &mut String, settings: &Settings, local_size: usize) {
    use std::fmt::Write;
    let mut header = String::new();
    let version = match shader_version() {
        Ok(v) => v,
        Err(e) => {
            println!("Couldn't get shader version: {}", e);
            "430".to_string()
        }
    };
    writeln!(&mut header, "#version {}", version).unwrap();
    writeln!(&mut header, "layout(local_size_x = {}) in;", local_size).unwrap();
    for value in &settings.values {
        if let Some(header_item) = value.format_glsl() {
            writeln!(&mut header, "{}", header_item).unwrap();
        }
    }
    *original_source = original_source.replace("#version 430", &header);
}

const PARSE: &str = concat!(
    "(?m)^ *(",
    r"uniform *(?P<kinduint>uint) (?P<nameuint>[a-zA-Z0-9_]+) *; *// *(?P<valueuint>([-+]?[0-9]+))|",
    r"uniform *(?P<kindfloat>float) (?P<namefloat>[a-zA-Z0-9_]+) *; *// *(?P<valuefloat>[-+]?[0-9]+(?:\.[0-9]+)?) +(?P<changefloat>[-+]?[0-9]+(?:\.[0-9]+)?)?|",
    r"uniform *(?P<kindvec3>vec3) (?P<namevec3>[a-zA-Z0-9_]+) *; *// *(?P<valuevec3>([-+]?[0-9]+(?:\.[0-9]+)? +){3}) *(?P<changevec3>[-+]?[0-9]+(?:\.[0-9]+)?)?",
    ") *\r?$"
);

fn settings_from_str(src: &str) -> Settings {
    lazy_static! {
        static ref RE: Regex = Regex::new(PARSE).expect("Failed to create regex");
        static ref DEFINES: Regex = Regex::new(r#"(?m)^ *#ifdef +(?P<name>[a-zA-Z0-9_]+) *\r?$"#)
            .expect("Failed to create regex");
    }
    let mut result = Settings::new();
    let mut once = false;
    for cap in RE.captures_iter(src) {
        once = true;
        let (name, setting) = if cap.name("kinduint").is_some() {
            let value = cap["valueuint"].parse().expect("Failed to extract regex");
            (&cap["nameuint"], SettingValueEnum::Int(value))
        } else if cap.name("kindfloat").is_some() {
            let value = cap["valuefloat"].parse().expect("Failed to extract regex");
            let change = cap["changefloat"].parse().expect("Failed to extract regex");
            (&cap["namefloat"], SettingValueEnum::Float(value, change))
        } else if cap.name("kindvec3").is_some() {
            let value = parse_vector3(&cap["valuevec3"]).expect("Failed to extract regex");
            let change = cap["changevec3"].parse().expect("Failed to extract regex");
            (&cap["namevec3"], SettingValueEnum::Vec3(value, change))
        } else {
            panic!("Regex returned invalid kind");
        };
        result
            .values
            .push(SettingValue::new(name.to_string(), setting));
    }
    assert!(once, "Regex should get at least one setting");
    result.values.push(SettingValue::new(
        "render_scale".to_string(),
        SettingValueEnum::Int(1),
    ));
    for cap in DEFINES.captures_iter(src) {
        let name = cap["name"].to_string();
        if result.get(&name) == None {
            let new_value = SettingValueEnum::Define(false);
            result.values.push(SettingValue::new(name, new_value));
        }
    }
    result
}

fn dump_binary(program: GLuint, path: impl AsRef<Path>) -> Result<(), Error> {
    unsafe {
        let mut program_binary_length = 0;
        gl::GetProgramiv(
            program,
            gl::PROGRAM_BINARY_LENGTH,
            &mut program_binary_length,
        );
        check_gl()?;
        let mut binary = vec![0u8; program_binary_length as usize];
        let mut length = 0;
        let mut binary_format = 0;
        gl::GetProgramBinary(
            program,
            program_binary_length,
            &mut length,
            &mut binary_format,
            binary.as_mut_ptr() as *mut c_void,
        );
        check_gl()?;
        if program_binary_length != length {
            println!(
                "Warning: GL_PROGRAM_BINARY_LENGTH ({}) didn't match glGetProgramBinary length ({})",
                program_binary_length, length
            );
        }
        File::create(path)?.write_all(&binary[0..(length as usize)])?;
        println!("Wrote binary (of format 0x{:x})", binary_format);
    }
    Ok(())
}
