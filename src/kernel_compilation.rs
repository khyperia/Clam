use crate::setting_value::SettingValueEnum;
use crate::settings::Settings;
use failure::Error;
use ocl::enums::ProgramBuildInfo;
use ocl::enums::ProgramBuildInfoResult;
use ocl::enums::ProgramInfo;
use ocl::enums::ProgramInfoResult;
use ocl::Buffer;
use ocl::Image;
use ocl::Kernel;
use ocl::OclPrm;
use ocl::Program;
use ocl::Queue;
use regex::Regex;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::thread;
use std::time::Duration;
use std::time::SystemTime;

const MANDELBOX: &str = include_str!("mandelbox.cl");
const MANDELBOX_PATH: &str = "src/mandelbox.cl";

fn dump_binary(program: &Program) -> Result<(), Error> {
    if let Ok(path) = env::var("CLAM5_BINARY") {
        if let ProgramInfoResult::Binaries(binaries) = program.info(ProgramInfo::Binaries)? {
            if binaries.len() != 1 {
                for (i, binary) in binaries.iter().enumerate() {
                    let this_path = format!("{}.{}", &path, i);
                    println!("Dumped binary: {}", this_path);
                    let mut file = File::create(this_path)?;
                    file.write_all(&binary[..])?;
                }
            } else {
                let mut file = File::create(path.to_string())?;
                file.write_all(&binaries[0][..])?;
                println!("Dumped binary: {}", path);
            }
        }
    }
    Ok(())
}

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

fn generate_src(settings: &Settings) -> String {
    let mut result = String::new();
    result.push_str("#define NOT_EDITOR 1\n");
    result.push_str("struct MandelboxCfg {\n");
    for value in &settings.values {
        if let Some(ref tmp) = value.format_opencl_struct() {
            result.push_str(tmp);
        }
    }
    result.push_str("};\n");
    for value in &settings.values {
        result.push_str(&value.format_opencl());
    }
    result
}

pub fn refresh_settings(settings: &mut Settings) -> Result<(), Error> {
    let src = get_src()?;
    set_src(settings, &src);
    Ok(())
}

pub fn rebuild<T: OclPrm>(
    queue: &Queue,
    texture: &Image<T>,
    settings: &mut Settings,
) -> Result<Kernel, Error> {
    let program = {
        let mut builder = Program::builder();
        let src = get_src()?;
        set_src(settings, &src);
        builder.source(generate_src(settings));
        builder.source(src);
        builder.devices(queue.device());
        builder.cmplr_opt("-cl-fast-relaxed-math");
        let device_name = queue.device().name()?;
        if device_name.contains("GeForce") {
            builder.cmplr_opt("-cl-nv-verbose");
        }
        builder.build(&queue.context())?
    };
    if let ProgramBuildInfoResult::BuildLog(log) =
        program.build_info(queue.device(), ProgramBuildInfo::BuildLog)?
    {
        let log = log.trim();
        if !log.is_empty() {
            println!("{}", log);
        }
    }
    dump_binary(&program)?;
    let kernel = Kernel::builder()
        .program(&program)
        .name("Main")
        .arg(texture)
        .arg(None::<&Buffer<f32>>)
        .arg(None::<&Buffer<u8>>)
        .arg(0u32)
        .arg(0u32)
        .arg(0u32)
        .build()?;
    Ok(kernel)
}

fn set_src(settings: &mut Settings, src: &str) {
    lazy_static! {
        static ref RE: Regex = Regex::new(
           r#"(?m)^ *extern *(?P<kind>float|int) (?P<name>[a-zA-Z0-9_]+)\([^)]*\); *// *(?P<value>[-+]?\d+(?:\.\d+)?) *(?P<change>[-+]?\d+(?:\.\d+)?)? *(?P<const>const)? *\r?$"#).unwrap();
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
                let value = cap["value"].parse().unwrap();
                let change = cap["change"].parse().unwrap();
                SettingValueEnum::F32(value, change)
            }
            "int" => {
                let value = cap["value"].parse().unwrap();
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
        static ref RE: Regex =
            Regex::new(r#"(?m)^ *#ifdef +(?P<name>[a-zA-Z0-9_]+) *\r?$"#).unwrap();
    }
    for cap in RE.captures_iter(src) {
        let name = &cap["name"];
        let new_value = SettingValueEnum::Define(false);
        variables.remove(name);
        settings.define_variable(name, new_value, false);
    }
}
