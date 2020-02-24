use crate::{
    check_gl, parse_vector3,
    setting_value::{SettingValue, SettingValueEnum},
    settings::Settings,
};
use failure::{err_msg, Error};
use gl::types::*;
use khygl::create_compute_program;
use lazy_static::lazy_static;
use regex::Regex;
use std::{
    ffi::CStr,
    fmt::Write,
    fs,
    fs::File,
    io::prelude::*,
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
            Err(_) => {
                println!("Warning: {} not found", self.path);
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

    pub fn rebuild(&self, settings: &Settings, local_size: usize) -> Result<GLuint, Error> {
        let mut to_compile = self.source.clone();
        generate_src(&mut to_compile, settings, local_size);
        create_compute_program(&[&to_compile])
    }
}

fn shader_version() -> Result<String, Error> {
    let ver = unsafe { CStr::from_ptr(gl::GetString(gl::SHADING_LANGUAGE_VERSION) as _) };
    check_gl()?;
    let ver = ver
        .to_str()?
        .split_whitespace()
        .next()
        .ok_or_else(|| err_msg("GL_SHADING_LANGUAGE_VERSION is empty"))?
        .replace(".", "");
    Ok(ver)
}

fn generate_src(original_source: &mut String, settings: &Settings, local_size: usize) {
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
        if let Some(header_item) = value.format_glsl(original_source) {
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
    ") *(?P<const>const)? *\r?$"
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
        result.values.push(SettingValue::new(
            name.to_string(),
            setting,
            cap.name("const").is_some(),
        ));
    }
    assert!(once, "Regex should get at least one setting");
    result.values.push(SettingValue::new(
        "render_scale".to_string(),
        SettingValueEnum::Int(1),
        false,
    ));
    for cap in DEFINES.captures_iter(src) {
        let name = cap["name"].to_string();
        let new_value = SettingValueEnum::Define(false);
        result
            .values
            .push(SettingValue::new(name, new_value, false));
    }
    result
}
