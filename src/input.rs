use failure::Error;
use sdl2::keyboard::Scancode as Key;
use settings::{SettingValue, Settings};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Default)]
pub struct Input {
    pressed_keys: HashMap<Key, Instant>,
    pub index: usize,
}

impl Input {
    pub fn new() -> Input {
        Input::default()
    }

    fn help() {
        println!("Keybindings:");
        // free:
        // QE
        // G
        //
        println!("WASD, [space]Z, IJKL, OU: move camera");
        println!("RF: focal distance/move speed");
        println!("NM: field of view");
        println!("Y: Write settings to disk. P: Read settings. V: Write keyframe.");
        println!("up/down/left/right: Adjust settings. T: Toggle zero setting.");
        println!("C: Toggle constant. B: Recompile kernel with new constants.");
        println!("X: Copy position to lightsource position");
        println!("H: Print this message");
    }

    pub fn key_down(&mut self, key: Key, settings: &mut Settings) {
        let time = Instant::now();
        let new_key = if let Entry::Vacant(entry) = self.pressed_keys.entry(key) {
            entry.insert(time);
            true
        } else {
            false
        };
        if new_key {
            self.run(settings, time);
        }
        match self.run_down(key, settings) {
            Ok(()) => (),
            Err(err) => println!("Error handling key down event: {}", err),
        }
    }

    pub fn key_up(&mut self, key: Key, settings: &mut Settings) {
        let time = Instant::now();
        if self.pressed_keys.contains_key(&key) {
            self.run(settings, time);
            self.pressed_keys.remove(&key);
        }
    }

    pub fn integrate(&mut self, settings: &mut Settings) {
        let now = Instant::now();
        self.run(settings, now);
    }

    fn run_down(&mut self, key: Key, settings: &mut Settings) -> Result<(), Error> {
        match key {
            Key::H => {
                Self::help();
            }
            Key::P => {
                settings.load("settings.clam5")?;
                println!("Settings loaded");
            }
            Key::Y => {
                settings.save("settings.clam5")?;
                println!("Settings saved");
            }
            Key::V => {
                settings.save_keyframe("keyframes.clam5")?;
                println!("Keyframe saved");
            }
            Key::Up => {
                if self.index == 0 {
                    self.index = settings.value_map().len() - 1;
                } else {
                    self.index -= 1;
                }
            }
            Key::Down => {
                self.index += 1;
                if self.index >= settings.value_map().len() {
                    self.index = 0;
                }
            }
            Key::Left => {
                let key = settings.nth(self.index);
                if let SettingValue::U32(value) = *settings.get(key).unwrap() {
                    if value != 0 {
                        settings.insert(key.into(), SettingValue::U32(value - 1));
                    }
                }
            }
            Key::Right => {
                let key = settings.nth(self.index);
                if let SettingValue::U32(value) = *settings.get(key).unwrap() {
                    settings.insert(key.into(), SettingValue::U32(value + 1));
                }
            }
            Key::T => {
                let key = settings.nth(self.index);
                let default = Settings::default_for(key).unwrap();
                let new_value = match *settings.get(key).unwrap() {
                    SettingValue::U32(value) => {
                        if value == 0 {
                            default
                        } else {
                            SettingValue::U32(0)
                        }
                    }
                    SettingValue::F32(value, speed) => {
                        if value == 0.0 {
                            default
                        } else {
                            SettingValue::F32(0.0, speed)
                        }
                    }
                };
                settings.insert(key.to_string(), new_value);
            }
            Key::C => {
                let key = settings.nth(self.index);
                let is_const = settings.is_const(key);
                settings.set_const(key, !is_const);
            }
            Key::B => {
                settings.rebuild();
            }
            Key::X => {
                let x = *settings.get("pos_x").unwrap();
                let y = *settings.get("pos_y").unwrap();
                let z = *settings.get("pos_z").unwrap();
                settings.insert("light_pos_1_x".to_string(), x);
                settings.insert("light_pos_1_y".to_string(), y);
                settings.insert("light_pos_1_z".to_string(), z);
            }
            _ => (),
        }
        Ok(())
    }

    fn run(&mut self, settings: &mut Settings, now: Instant) {
        self.camera_3d(settings, now);
        self.exp_setting(settings, now, "focal_distance".into(), Key::R, Key::F);
        self.exp_setting(settings, now, "fov".into(), Key::N, Key::M);
        self.manual_control(settings, now);
        for value in self.pressed_keys.values_mut() {
            *value = now;
        }
    }

    fn is_pressed(&self, now: Instant, key: Key) -> Option<f32> {
        if let Some(&old) = self.pressed_keys.get(&key) {
            let dt = now.duration_since(old);
            let flt = dt.as_secs() as f32 + dt.subsec_nanos() as f32 * 1e-9;
            Some(flt)
        } else {
            None
        }
    }

    fn get_f32(settings: &Settings, key: &str) -> Option<f32> {
        match settings.get(key) {
            Some(&SettingValue::F32(val, _)) => Some(val),
            _ => None,
        }
    }

    fn camera_3d(&self, settings: &mut Settings, now: Instant) {
        let move_speed = Self::get_f32(settings, "focal_distance").unwrap() * 0.5;
        let turn_speed = Self::get_f32(settings, "fov").unwrap();
        let roll_speed = 1.0;
        let mut pos = Vector::read(settings, "pos_x", "pos_y", "pos_z").unwrap();
        let mut look = Vector::read(settings, "look_x", "look_y", "look_z").unwrap();
        let mut up = Vector::read(settings, "up_x", "up_y", "up_z").unwrap();
        let old = (pos, look, up);
        let right = Vector::cross(look, up);
        if let Some(dt) = self.is_pressed(now, Key::W) {
            pos = pos + look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::S) {
            pos = pos - look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::D) {
            pos = pos + right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::A) {
            pos = pos - right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Space) {
            pos = pos + up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Z) {
            pos = pos - up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::I) {
            look = look.rotate(up, turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::K) {
            look = look.rotate(up, -turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::L) {
            look = look.rotate(right, turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::J) {
            look = look.rotate(right, -turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::O) {
            up = up.rotate(right, roll_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::U) {
            up = up.rotate(right, -roll_speed * dt);
        }
        if old != (pos, look, up) {
            look = look.normalized();
            up = Vector::cross(Vector::cross(look, up), look).normalized();
            pos.write(settings, "pos_x", "pos_y", "pos_z");
            look.write(settings, "look_x", "look_y", "look_z");
            up.write(settings, "up_x", "up_y", "up_z");
        }
    }

    fn exp_setting(
        &self,
        settings: &mut Settings,
        now: Instant,
        key: String,
        increase: Key,
        decrease: Key,
    ) {
        let (mut value, change) = match *settings.get(&key).unwrap() {
            SettingValue::F32(value, change) => (value, -change + 1.0),
            _ => return,
        };
        if let Some(dt) = self.is_pressed(now, increase) {
            value *= change.powf(dt);
        }
        if let Some(dt) = self.is_pressed(now, decrease) {
            value *= change.powf(-dt);
        }
        settings.insert(key, SettingValue::F32(value, -change + 1.0));
    }

    fn manual_control(&mut self, settings: &mut Settings, now: Instant) {
        let mut do_control = |dt| {
            let key = settings.nth(self.index);
            if let SettingValue::F32(value, change) = *settings.get(key).unwrap() {
                if change < 0.0 {
                    settings.insert(
                        key.into(),
                        SettingValue::F32(value * (-change + 1.0).powf(dt), change),
                    );
                } else {
                    settings.insert(key.into(), SettingValue::F32(value + dt * change, change));
                }
            };
        };
        if let Some(dt) = self.is_pressed(now, Key::Left) {
            do_control(-dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Right) {
            do_control(dt);
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector {
    pub fn new(x: f32, y: f32, z: f32) -> Vector {
        Vector { x, y, z }
    }

    fn read(settings: &Settings, x: &str, y: &str, z: &str) -> Option<Vector> {
        match (settings.get(x), settings.get(y), settings.get(z)) {
            (
                Some(&SettingValue::F32(x, _)),
                Some(&SettingValue::F32(y, _)),
                Some(&SettingValue::F32(z, _)),
            ) => Some(Self::new(x, y, z)),
            _ => None,
        }
    }

    fn write(&self, settings: &mut Settings, x: &str, y: &str, z: &str) {
        settings.insert(x.into(), SettingValue::F32(self.x, 1.0));
        settings.insert(y.into(), SettingValue::F32(self.y, 1.0));
        settings.insert(z.into(), SettingValue::F32(self.z, 1.0));
    }

    pub fn len2(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn len(&self) -> f32 {
        self.len2().sqrt()
    }

    pub fn normalized(&self) -> Vector {
        *self * (1.0 / self.len())
    }

    pub fn cross(l: Vector, r: Vector) -> Vector {
        Self::new(
            l.y * r.z - l.z * r.y,
            l.z * r.x - l.x * r.z,
            l.x * r.y - l.y * r.x,
        )
    }

    pub fn rotate(&self, direction: Vector, amount: f32) -> Vector {
        (*self + direction * amount).normalized()
    }
}

impl ::std::ops::Add for Vector {
    type Output = Vector;

    fn add(self, r: Vector) -> Self::Output {
        Self::new(self.x + r.x, self.y + r.y, self.z + r.z)
    }
}

impl ::std::ops::Sub for Vector {
    type Output = Vector;

    fn sub(self, r: Vector) -> Self::Output {
        Self::new(self.x - r.x, self.y - r.y, self.z - r.z)
    }
}

impl ::std::ops::Mul<f32> for Vector {
    type Output = Vector;

    fn mul(self, r: f32) -> Self::Output {
        Self::new(self.x * r, self.y * r, self.z * r)
    }
}
