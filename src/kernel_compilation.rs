use crate::{check_gl, setting_value::SettingValueEnum, settings::Settings};
use failure::{err_msg, Error};
use gl::types::*;
use khygl::create_compute_program;
use lazy_static::lazy_static;
use regex::Regex;
use std::{
    collections::HashSet,
    ffi::CStr,
    fmt::Write,
    fs,
    fs::File,
    io::prelude::*,
    thread,
    time::{Duration, SystemTime},
};

const MANDELBOX: &str = include_str!("mandelbox.comp.glsl");
const MANDELBOX_PATH: &str = "src/mandelbox.comp.glsl";

// the "notify" crate is broken af, so roll our own
pub fn watch_src<F: Fn() + Send + 'static>(on_changed: F) {
    thread::spawn(move || {
        fn get_time() -> Option<SystemTime> {
            let meta = match fs::metadata(MANDELBOX_PATH) {
                Ok(v) => v,
                Err(_) => return None,
            };
            let time = match meta.modified() {
                Ok(v) => v,
                Err(_) => return None,
            };
            Some(time)
        }
        let mut time = get_time();
        loop {
            thread::sleep(Duration::from_secs(1));
            let new_time = get_time();
            if new_time != time {
                time = new_time;
                on_changed();
            }
        }
    });
}

fn get_src() -> Result<String, Error> {
    let mut file = match File::open(MANDELBOX_PATH) {
        Ok(f) => f,
        Err(_) => {
            println!("Warning: {} not found", MANDELBOX_PATH);
            return Ok(MANDELBOX.to_string());
        }
    };
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
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

pub fn refresh_settings(settings: &mut Settings) -> Result<(), Error> {
    let src = get_src()?;
    set_src(settings, &src);
    Ok(())
}

pub fn rebuild(settings: &mut Settings, local_size: usize) -> Result<GLuint, Error> {
    let mut src = get_src()?;
    set_src(settings, &src);
    generate_src(&mut src, settings, local_size);
    create_compute_program(&[&src])
}

fn set_src(settings: &mut Settings, src: &str) {
    lazy_static! {
        static ref RE: Regex = Regex::new(
           r#"(?m)^ *uniform *(?P<kind>float|uint) (?P<name>[a-zA-Z0-9_]+) *; *// *(?P<value>[-+]?[0-9]+(?:\.[0-9]+)?) *(?P<change>[-+]?[0-9]+(?:\.[0-9]+)?)? *(?P<const>const)? *\r?$"#).expect("Failed to create regex");
    }
    let mut set: HashSet<String> = settings
        .values
        .iter()
        .map(|x| x.key().to_string())
        .collect();
    let mut once = false;
    for cap in RE.captures_iter(src) {
        once = true;
        let kind = &cap["kind"];
        let name = &cap["name"];
        set.remove(name);
        let setting = match kind {
            "float" => {
                let value = cap["value"].parse().expect("Failed to extract regex");
                let change = cap["change"].parse().expect("Failed to extract regex");
                SettingValueEnum::F32(value, change)
            }
            "uint" => {
                let value = cap["value"].parse().expect("Failed to extract regex");
                SettingValueEnum::U32(value)
            }
            _ => {
                panic!("Regex returned invalid kind");
            }
        };
        settings.define_variable(name, setting, cap.name("const").is_some());
    }
    settings.define_variable("render_scale", SettingValueEnum::U32(1), false);
    set.remove("render_scale");
    assert!(once, "Regex should get at least one setting");
    find_defines(settings, src, &mut set);
    for to_delete in set {
        settings.delete_variable(&to_delete);
    }
}

fn find_defines(settings: &mut Settings, src: &str, variables: &mut HashSet<String>) {
    lazy_static! {
        static ref RE: Regex = Regex::new(r#"(?m)^ *#ifdef +(?P<name>[a-zA-Z0-9_]+) *\r?$"#)
            .expect("Failed to create regex");
    }
    for cap in RE.captures_iter(src) {
        let name = &cap["name"];
        let new_value = SettingValueEnum::Define(false);
        variables.remove(name);
        settings.define_variable(name, new_value, false);
    }
}
