use std::collections::HashMap;
use std::error::Error;
use std::io::Write;
use std::io::BufRead;

pub enum SettingValue {
    U32(u32),
    F32(f32),
}

pub type Settings = HashMap<String, SettingValue>;

pub fn save_settings(settings: &Settings, file: &str) -> Result<(), Box<Error>> {
    let file = ::std::fs::File::create(file)?;
    let mut writer = ::std::io::BufWriter::new(&file);
    for (key, value) in settings {
        match *value {
            SettingValue::F32(value) => writeln!(&mut writer, "{} = {}", key, value)?,
            SettingValue::U32(value) => writeln!(&mut writer, "{} = {}", key, value)?,
        }
    }
    Ok(())
}

pub fn load_settings(settings: &mut Settings, file: &str) -> Result<(), Box<Error>> {
    let file = ::std::fs::File::open(file)?;
    let reader = ::std::io::BufReader::new(&file);
    for line in reader.lines() {
        let line = line?;
        let split = line.rsplitn(2, '=').collect::<Vec<_>>();
        let key = split[1].trim();
        let value = split[0].trim();
        match settings[key] {
            SettingValue::F32(_) => settings.insert(key.into(), SettingValue::F32(value.parse()?)),
            SettingValue::U32(_) => settings.insert(key.into(), SettingValue::U32(value.parse()?)),
        };
    }
    Ok(())
}
