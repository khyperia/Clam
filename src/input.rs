use glium::glutin::VirtualKeyCode as Key;
use settings::*;
use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;

pub struct Input {
    pressed_keys: HashMap<Key, Instant>,
}

impl Input {
    pub fn new() -> Input {
        Input { pressed_keys: HashMap::new() }
    }

    pub fn key_down(&mut self, key: Key, time: Instant, settings: &mut Settings) {
        if !self.pressed_keys.contains_key(&key) {
            self.pressed_keys.insert(key, time);
            self.run(settings, time);
            match self.run_down(key, settings) {
                Ok(()) => (),
                Err(err) => println!("Error handing key down event: {}", err),
            }
        }
    }

    pub fn key_up(&mut self, key: Key, time: Instant, settings: &mut Settings) {
        if self.pressed_keys.contains_key(&key) {
            self.run(settings, time);
            self.pressed_keys.remove(&key);
        }
    }

    pub fn integrate(&mut self, settings: &mut Settings) {
        let now = Instant::now();
        self.run(settings, now);
    }

    fn run_down(&mut self, key: Key, settings: &mut Settings) -> Result<(), Box<Error>> {
        match key {
            Key::T => {
                load_settings(settings, "settings.clam5")?;
                println!("Settings loaded");
            }
            Key::P => {
                save_settings(settings, "settings.clam5")?;
                println!("Settings saved");
            }
            _ => (),
        }
        Ok(())
    }

    fn run(&mut self, settings: &mut Settings, now: Instant) {
        self.camera_3d(settings, now);
        self.exp_setting(settings, now, "focal_distance".into(), Key::R, Key::F, 2.0);
        self.exp_setting(settings, now, "fov".into(), Key::N, Key::M, 2.0);
        for value in self.pressed_keys.values_mut() {
            *value = now;
        }
    }

    fn is_pressed(&self, now: Instant, key: Key) -> Option<f32> {
        if let Some(&old) = self.pressed_keys.get(&key) {
            let (dt, sign) = if now < old {
                (old.duration_since(now), true)
            } else {
                (now.duration_since(old), false)
            };
            let flt = dt.as_secs() as f32 + dt.subsec_nanos() as f32 * 1e-9;
            Some(if sign { -flt } else { flt })
        } else {
            None
        }
    }

    fn get_f32(settings: &Settings, key: &str) -> Option<f32> {
        match settings.get(key) {
            Some(&SettingValue::F32(val)) => Some(val),
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
        scale: f32,
    ) {
        let mut value = Self::get_f32(settings, &key).unwrap();
        if let Some(dt) = self.is_pressed(now, increase) {
            value *= scale.powf(dt);
        }
        if let Some(dt) = self.is_pressed(now, decrease) {
            value *= scale.powf(-dt);
        }
        settings.insert(key, SettingValue::F32(value));
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
        Vector { x: x, y: y, z: z }
    }

    fn read(settings: &Settings, x: &str, y: &str, z: &str) -> Option<Vector> {
        match (settings.get(x), settings.get(y), settings.get(z)) {
            (Some(&SettingValue::F32(x)),
             Some(&SettingValue::F32(y)),
             Some(&SettingValue::F32(z))) => Some(Self::new(x, y, z)),
            _ => None,
        }
    }

    fn write(&self, settings: &mut Settings, x: &str, y: &str, z: &str) {
        settings.insert(x.into(), SettingValue::F32(self.x));
        settings.insert(y.into(), SettingValue::F32(self.y));
        settings.insert(z.into(), SettingValue::F32(self.z));
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
