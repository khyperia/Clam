use crate::{
    kernel_compilation::RealizedSource,
    settings::{KeyframeList, Settings},
    Key,
};
use cgmath::{prelude::*, Quaternion, Rad, Vector3};
use failure::Error;
use std::{
    collections::{hash_map::Entry, HashMap},
    time::Instant,
};

#[derive(Default)]
pub struct Input {
    pressed_keys: HashMap<Key, Instant>,
    spaceship: Option<(Vector3<f64>, Vector3<f64>, Instant)>,
    video: Option<KeyframeList>,
    cur_video: usize,
    video_len: usize,
    pub index: usize,
}

impl Input {
    pub fn new() -> Self {
        Self::default()
    }

    fn help() {
        println!("Keybindings:");
        // free:
        // QE
        //
        //
        println!("WASD, [space]Z, IJKL, OU: move camera");
        println!("RF: focal distance/move speed");
        println!("NM: field of view");
        println!(
            "Y: Write settings to disk. P: Read settings. V: Write keyframe. G: Play keyframes."
        );
        println!("up/down/left/right: Adjust settings. T: Toggle zero setting.");
        println!("C: Toggle constant. B: Recompile kernel with new constants.");
        println!("X: Copy position to lightsource position");
        println!("`: Spaceship!");
        println!("H: Print this message");
    }

    pub fn key_down(
        &mut self,
        key: Key,
        settings: &mut Settings,
        realized_source: &RealizedSource,
    ) {
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
        match self.run_down(key, settings, realized_source) {
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

    fn run_down(
        &mut self,
        key: Key,
        settings: &mut Settings,
        realized_source: &RealizedSource,
    ) -> Result<(), Error> {
        match key {
            Key::H => {
                Self::help();
            }
            Key::P => {
                *settings = Settings::load("settings.clam5", settings)?;
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
            Key::G => match KeyframeList::new("keyframes.clam5", realized_source) {
                Ok(ok) => {
                    self.cur_video = 0;
                    self.video_len = ok.len() * 100;
                    self.video = Some(ok);
                    println!("Playing video")
                }
                Err(err) => println!("Failed to open keyframes.clam5: {}", err),
            },
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
            Key::X => {
                let pos = settings.find("pos").value().clone();
                settings.find_mut("light_pos_1").set_value(pos);
            }
            Key::Grave => {
                if self.spaceship.is_none() {
                    self.spaceship = Some((Vector3::zero(), Vector3::zero(), Instant::now()));
                } else {
                    self.spaceship = None;
                }
            }
            Key::Key1 => {
                unsafe {
                    gl::Enable(gl::FRAMEBUFFER_SRGB);
                }
                crate::check_gl()?;
                *settings.find_mut("gamma").unwrap_float_mut() = 1.0;
            }
            Key::Key2 => {
                unsafe {
                    gl::Disable(gl::FRAMEBUFFER_SRGB);
                }
                crate::check_gl()?;
                *settings.find_mut("gamma").unwrap_float_mut() = 0.0;
            }
            _ => (),
        }
        Ok(())
    }

    fn run(&mut self, settings: &mut Settings, now: Instant) {
        if self.spaceship.is_some() {
            self.spaceship(settings, now);
        } else {
            self.camera_3d(settings, now);
        }
        self.exp_setting(
            settings,
            now,
            "focal_distance",
            settings.find("fov").unwrap_float(),
            Key::R,
            Key::F,
        );
        self.exp_setting(settings, now, "fov", 1.0, Key::N, Key::M);
        self.manual_control(settings, now);
        for value in self.pressed_keys.values_mut() {
            *value = now;
        }
        if let Some(ref mut video) = self.video {
            *settings = video.interpolate(self.cur_video as f64 / self.video_len as f64, false);
            self.cur_video += 1;
            if self.cur_video > self.video_len {
                // TODO
                self.video = None;
            }
        }
    }

    fn is_pressed(&self, now: Instant, key: Key) -> Option<f64> {
        if let Some(&old) = self.pressed_keys.get(&key) {
            Some(now.duration_since(old).as_secs_f64())
        } else {
            None
        }
    }

    fn camera_3d(&self, settings: &mut Settings, now: Instant) {
        let move_speed = settings.find("focal_distance").unwrap_float() * 0.5;
        let turn_speed = settings.find("fov").unwrap_float();
        let roll_speed = 1.0;
        let mut pos = settings.find("pos").unwrap_vec3();
        let mut look = settings.find("look").unwrap_vec3();
        let mut up = settings.find("up").unwrap_vec3();
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
            *settings.find_mut("pos").unwrap_vec3_mut() = pos;
            *settings.find_mut("look").unwrap_vec3_mut() = look;
            *settings.find_mut("up").unwrap_vec3_mut() = up;
        }
    }

    fn exp_setting(
        &self,
        settings: &mut Settings,
        now: Instant,
        key: &str,
        mul: f64,
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

    fn spaceship(&mut self, settings: &mut Settings, now: Instant) {
        let move_speed = settings.find("focal_distance").unwrap_float() / 16.0;
        let turn_speed = settings.find("fov").unwrap_float() / 2.0;
        let roll_speed = 1.0 / 4.0;
        let mut look = settings.find("look").unwrap_vec3();
        let mut up = settings.find("up").unwrap_vec3();
        let right = Vector3::cross(look, up);
        let mut thrust = Vector3::zero();
        let mut angular_thrust = Vector3::zero();
        if let Some(dt) = self.is_pressed(now, Key::W) {
            thrust += look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::S) {
            thrust -= look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::D) {
            thrust += right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::A) {
            thrust -= right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Space) {
            thrust += up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Z) {
            thrust -= up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::I) {
            angular_thrust += right * (turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::K) {
            angular_thrust += right * (-turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::L) {
            angular_thrust += up * (-turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::J) {
            angular_thrust += up * (turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::O) {
            angular_thrust += look * (roll_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::U) {
            angular_thrust += look * (-roll_speed * dt);
        }

        let (velocity, angularvel, last_update) = self.spaceship.as_mut().unwrap();
        let dt = now.duration_since(*last_update).as_secs_f64();
        *last_update = now;
        *velocity += thrust;
        *angularvel += angular_thrust;

        let mag = angularvel.magnitude() * dt;
        if mag > 0.0 {
            let roll = Quaternion::from_axis_angle(angularvel.normalize(), Rad(mag));
            look = roll * look;
            up = roll * up;
        }

        let mut pos = settings.find("pos").unwrap_vec3();
        pos += *velocity * dt;

        look = look.normalize();
        up = Vector3::cross(Vector3::cross(look, up), look).normalize();
        *settings.find_mut("pos").unwrap_vec3_mut() = pos;
        *settings.find_mut("look").unwrap_vec3_mut() = look;
        *settings.find_mut("up").unwrap_vec3_mut() = up;
    }
}
