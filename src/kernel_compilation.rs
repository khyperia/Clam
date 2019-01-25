use display::ScreenEvent;
use failure::Error;
use mandelbox_cfg::MandelboxCfg;
use settings::{SettingValue, Settings};
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

const MANDELBOX: &str = include_str!("mandelbox.cl");
const MANDELBOX_PATH: &str = "src/mandelbox.cl";

fn dump_binary(program: &ocl::Program) -> Result<(), Error> {
    if let Ok(path) = env::var("CLAM5_BINARY") {
        if let ocl::enums::ProgramInfoResult::Binaries(binaries) =
            program.info(ocl::enums::ProgramInfo::Binaries)?
        {
            if binaries.len() != 1 {
                for (i, binary) in binaries.iter().enumerate() {
                    let mut file = File::create(format!("{}.{}", &path, i))?;
                    file.write_all(&binary[..])?;
                }
            } else {
                let mut file = File::create(path.to_string())?;
                file.write_all(&binaries[0][..])?;
            }
            println!("Dumped binaries");
        }
    }
    Ok(())
}

fn generate_rust_struct(settings: &Settings) -> String {
    let mut result = String::new();
    result.push_str("pub struct MandelboxCfg {\n");
    for name in Settings::keys() {
        if name == "render_scale" {
            continue;
        }
        let value = settings.get(name).unwrap();
        match value {
            SettingValue::U32(_) => result.push_str(&format!("    {}: u32,\n", name)),
            SettingValue::F32(_, _) => result.push_str(&format!("    {}: f32,\n", name)),
        }
    }
    result.push_str("}\n");
    result
}

fn generate_header(settings: &Settings) -> String {
    let mut result = String::new();
    result.push_str("struct MandelboxCfg\n");
    result.push_str("{\n");
    for name in Settings::keys() {
        if name == "render_scale" {
            continue;
        }
        let value = settings.get(name).unwrap();
        match value {
            SettingValue::U32(_) => result.push_str(&format!("    int _{};\n", name)),
            SettingValue::F32(_, _) => result.push_str(&format!("    float _{};\n", name)),
        }
    }
    result.push_str("};\n\n");
    for name in Settings::keys() {
        if name == "render_scale" {
            continue;
        }
        result.push_str(&format!("#ifndef {}\n", name));
        result.push_str(&format!("#define {} cfg->_{}\n", name, name));
        result.push_str(&format!("#endif\n"));
    }
    result
}

pub fn check_header(settings: &Settings) -> bool {
    let header = generate_header(settings);
    if !MANDELBOX.starts_with(&header) {
        println!("Header check failed:");
        println!("{}", header);
        println!("-----");
        println!("{}", generate_rust_struct(settings));
        false
    } else {
        true
    }
}

// the "notify" crate is broken af, so roll our own
pub fn watch_src(sender: mpsc::Sender<ScreenEvent>) -> Result<(), Error> {
    thread::spawn(move || {
        fn get_time() -> Option<::std::time::SystemTime> {
            let meta = match ::std::fs::metadata(MANDELBOX_PATH) {
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
                match sender.send(ScreenEvent::KernelChanged) {
                    Ok(()) => (),
                    Err(mpsc::SendError(_)) => break,
                }
            }
        }
    });
    Ok(())
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

pub fn rebuild(queue: &ocl::Queue, settings: &Settings) -> Result<ocl::Kernel, Error> {
    let program = {
        let mut builder = ocl::Program::builder();
        builder.source(get_src()?);
        builder.devices(queue.device());
        builder.cmplr_opt("-cl-fast-relaxed-math");
        let device_name = queue.device().name()?;
        if device_name.contains("GeForce") {
            builder.cmplr_opt("-cl-nv-verbose");
        }
        for key in settings.constants() {
            let value = settings.get(&key).unwrap();
            match *value {
                SettingValue::F32(value, _) => {
                    builder.cmplr_opt(format!("-D {}={:.16}f", key, value))
                }
                SettingValue::U32(value) => builder.cmplr_opt(format!("-D {}={}", key, value)),
            };
        }
        builder.build(&queue.context())?
    };
    if let ocl::enums::ProgramBuildInfoResult::BuildLog(log) =
        program.build_info(queue.device(), ocl::enums::ProgramBuildInfo::BuildLog)?
    {
        let log = log.trim();
        if !log.is_empty() {
            println!("{}", log);
        }
    }
    dump_binary(&program)?;
    let kernel = ocl::Kernel::builder()
        .program(&program)
        .name("Main")
        .arg(None::<&ocl::Buffer<u8>>)
        .arg(None::<&ocl::Buffer<MandelboxCfg>>)
        .arg(0u32)
        .arg(0u32)
        .arg(0u32)
        .build()?;
    Ok(kernel)
}