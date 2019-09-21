use crate::{settings::Settings, Key};
use cgmath::{prelude::*, Quaternion, Rad, Vector3};
use failure::Error;
use std::{
    collections::{hash_map::Entry, HashMap},
    time::Instant,
};

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
                    self.index = settings.values.len() - 1;
                } else {
                    self.index -= 1;
                }
            }
            Key::Down => {
                self.index += 1;
                if self.index >= settings.values.len() {
                    self.index = 0;
                }
            }
            Key::Left => settings.values[self.index].change_one(false),
            Key::Right => settings.values[self.index].change_one(true),
            Key::T => settings.values[self.index].toggle(),
            Key::C => {
                let is_const = settings.values[self.index].is_const();
                settings.values[self.index].set_const(!is_const);
            }
            Key::B => settings.rebuild(),
            Key::X => {
                let x = settings.find("pos_x").clone();
                let y = settings.find("pos_y").clone();
                let z = settings.find("pos_z").clone();
                settings.find_mut("light_pos_1_x").set_value(*x.value());
                settings.find_mut("light_pos_1_y").set_value(*y.value());
                settings.find_mut("light_pos_1_z").set_value(*z.value());
            }
            _ => (),
        }
        Ok(())
    }

    fn run(&mut self, settings: &mut Settings, now: Instant) {
        self.camera_3d(settings, now);
        self.exp_setting(
            settings,
            now,
            "focal_distance",
            settings.find("fov").unwrap_f32(),
            Key::R,
            Key::F,
        );
        self.exp_setting(settings, now, "fov", 1.0, Key::N, Key::M);
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

    fn camera_3d(&self, settings: &mut Settings, now: Instant) {
        let move_speed = settings.find("focal_distance").unwrap_f32() * 0.5;
        let turn_speed = settings.find("fov").unwrap_f32();
        let roll_speed = 1.0;
        let mut pos = settings.read_vector("pos_x", "pos_y", "pos_z");
        let mut look = settings.read_vector("look_x", "look_y", "look_z");
        let mut up = settings.read_vector("up_x", "up_y", "up_z");
        let old = (pos, look, up);
        let right = Vector3::cross(look, up);
        if let Some(dt) = self.is_pressed(now, Key::W) {
            pos += look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::S) {
            pos -= look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::D) {
            pos += right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::A) {
            pos -= right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Space) {
            pos += up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Z) {
            pos -= up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::I) {
            look = Quaternion::from_axis_angle(right, Rad(turn_speed * dt)) * look;
        }
        if let Some(dt) = self.is_pressed(now, Key::K) {
            look = Quaternion::from_axis_angle(right, Rad(-turn_speed * dt)) * look;
        }
        if let Some(dt) = self.is_pressed(now, Key::L) {
            look = Quaternion::from_axis_angle(up, Rad(-turn_speed * dt)) * look;
        }
        if let Some(dt) = self.is_pressed(now, Key::J) {
            look = Quaternion::from_axis_angle(up, Rad(turn_speed * dt)) * look;
        }
        if let Some(dt) = self.is_pressed(now, Key::O) {
            up = Quaternion::from_axis_angle(look, Rad(roll_speed * dt)) * up;
        }
        if let Some(dt) = self.is_pressed(now, Key::U) {
            up = Quaternion::from_axis_angle(look, Rad(-roll_speed * dt)) * up;
        }
        if old != (pos, look, up) {
            look = look.normalize();
            up = Vector3::cross(Vector3::cross(look, up), look).normalize();
            settings.write_vector(pos, "pos_x", "pos_y", "pos_z");
            settings.write_vector(look, "look_x", "look_y", "look_z");
            settings.write_vector(up, "up_x", "up_y", "up_z");
        }
    }

    fn exp_setting(
        &self,
        settings: &mut Settings,
        now: Instant,
        key: &str,
        mul: f32,
        increase: Key,
        decrease: Key,
    ) {
        if let Some(dt) = self.is_pressed(now, increase) {
            settings.find_mut(key).change(true, dt * mul);
        }
        if let Some(dt) = self.is_pressed(now, decrease) {
            settings.find_mut(key).change(false, dt * mul);
        }
    }

    fn manual_control(&mut self, settings: &mut Settings, now: Instant) {
        if let Some(dt) = self.is_pressed(now, Key::Right) {
            settings.values[self.index].change(true, dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Left) {
            settings.values[self.index].change(false, dt);
        }
    }
}
