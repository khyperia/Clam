use crate::{
    keyframe_list::KeyframeList, settings::Settings, settings_input::SettingsInput, Error, Key,
};
use cgmath::{prelude::*, Quaternion, Rad, Vector3};
use instant::Instant;
use log::info;
use std::collections::{hash_map::Entry, HashMap};

pub struct Input {
    pressed_keys: HashMap<Key, Instant>,
    spaceship: Option<(Vector3<f64>, Vector3<f64>, Instant)>,
    cur_video_secs: f64,
    video_len_secs: f64,
    last_update: Instant,
    pub settings_input: SettingsInput,
}

impl Input {
    pub fn new() -> Self {
        Self {
            pressed_keys: HashMap::new(),
            spaceship: None,
            cur_video_secs: 0.0,
            video_len_secs: 0.0,
            last_update: Instant::now(),
            settings_input: SettingsInput::new(),
        }
    }

    fn help() {
        info!("Keybindings:");
        // free:
        // QE
        //
        // CB
        info!("WASD, [space]Z, IJKL, OU: move camera");
        info!("RF: focal distance/move speed");
        info!("NM: field of view");
        info!("Y: Write settings to disk. P: Read settings. V: Write keyframe. G: Play keyframes.");
        info!("up/down/left/right: Adjust settings. T: Toggle zero setting.");
        info!("X: Copy position to lightsource position");
        info!("`: Spaceship!");
        info!("H: Print this message");
    }

    pub fn key_down(
        &mut self,
        key: Key,
        settings: &mut Settings,
        default_settings: &Settings,
        keyframes: &mut KeyframeList,
    ) {
        let time = Instant::now();
        let new_key = if let Entry::Vacant(entry) = self.pressed_keys.entry(key) {
            entry.insert(time);
            true
        } else {
            false
        };
        if new_key {
            self.run(settings, keyframes, time);
        }
        match self.run_down(key, settings, default_settings, keyframes) {
            Ok(()) => (),
            Err(err) => info!("Error handling key down event: {}", err),
        }
    }

    pub fn key_up(&mut self, key: Key, settings: &mut Settings, keyframes: &KeyframeList) {
        let time = Instant::now();
        if self.pressed_keys.contains_key(&key) {
            self.run(settings, keyframes, time);
            self.pressed_keys.remove(&key);
        }
    }

    pub fn integrate(&mut self, settings: &mut Settings, keyframes: &KeyframeList) {
        let now = Instant::now();
        self.run(settings, keyframes, now);
    }

    fn run_down(
        &mut self,
        key: Key,
        settings: &mut Settings,
        default_settings: &Settings,
        keyframes: &mut KeyframeList,
    ) -> Result<(), Error> {
        match key {
            Key::KeyH => {
                Self::help();
            }
            Key::KeyP => {
                *settings = Settings::load("settings.clam5", settings)?;
                info!("Settings loaded");
            }
            Key::KeyY => {
                settings.save("settings.clam5", default_settings)?;
                info!("Settings saved");
            }
            Key::KeyV => {
                keyframes.push(settings.clone());
                keyframes.save("keyframes.clam5", default_settings)?;
                info!("Keyframe saved");
            }
            Key::KeyG => {
                self.cur_video_secs = 0.0;
                self.video_len_secs = keyframes.len() as f64 * (10.0 / 6.0);
                info!("Playing video")
            }
            Key::ArrowUp => self.settings_input.up_one(settings),
            Key::ArrowDown => self.settings_input.down_one(settings),
            Key::ArrowLeft => self.settings_input.left_one(settings),
            Key::ArrowRight => self.settings_input.right_one(settings),
            Key::KeyT => self.settings_input.toggle(settings),
            Key::KeyX => {
                let pos = settings.find("pos").value().clone();
                settings.find_mut("light_pos").set_value(pos);
            }
            Key::Backquote => {
                if self.spaceship.is_none() {
                    self.spaceship = Some((Vector3::zero(), Vector3::zero(), Instant::now()));
                } else {
                    self.spaceship = None;
                }
            }
            _ => (),
        }
        Ok(())
    }

    fn run(&mut self, settings: &mut Settings, keyframes: &KeyframeList, now: Instant) {
        let dt = (now - self.last_update).as_secs_f64();
        self.last_update = now;
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
            Key::KeyR,
            Key::KeyF,
        );
        self.exp_setting(settings, now, "fov", 1.0, Key::KeyN, Key::KeyM);
        self.manual_control(settings, now);
        for value in self.pressed_keys.values_mut() {
            *value = now;
        }
        if self.cur_video_secs < self.video_len_secs {
            *settings = keyframes.interpolate(self.cur_video_secs / self.video_len_secs, false);
            self.cur_video_secs += dt;
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
        if let Some(dt) = self.is_pressed(now, Key::KeyW) {
            pos += look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyS) {
            pos -= look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyD) {
            pos += right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyA) {
            pos -= right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Space) {
            pos += up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyZ) {
            pos -= up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyI) {
            look = Quaternion::from_axis_angle(right, Rad(turn_speed * dt)) * look;
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyK) {
            look = Quaternion::from_axis_angle(right, Rad(-turn_speed * dt)) * look;
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyL) {
            look = Quaternion::from_axis_angle(up, Rad(-turn_speed * dt)) * look;
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyJ) {
            look = Quaternion::from_axis_angle(up, Rad(turn_speed * dt)) * look;
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyO) {
            up = Quaternion::from_axis_angle(look, Rad(roll_speed * dt)) * up;
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyU) {
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
            settings.find_mut(key).change(0, true, dt * mul);
        }
        if let Some(dt) = self.is_pressed(now, decrease) {
            settings.find_mut(key).change(0, false, dt * mul);
        }
    }

    fn manual_control(&mut self, settings: &mut Settings, now: Instant) {
        if let Some(dt) = self.is_pressed(now, Key::ArrowRight) {
            self.settings_input.right_hold(settings, dt)
        }
        if let Some(dt) = self.is_pressed(now, Key::ArrowLeft) {
            self.settings_input.left_hold(settings, dt)
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
        if let Some(dt) = self.is_pressed(now, Key::KeyW) {
            thrust += look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyS) {
            thrust -= look * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyD) {
            thrust += right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyA) {
            thrust -= right * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::Space) {
            thrust += up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyZ) {
            thrust -= up * (move_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyI) {
            angular_thrust += right * (turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyK) {
            angular_thrust += right * (-turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyL) {
            angular_thrust += up * (-turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyJ) {
            angular_thrust += up * (turn_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyO) {
            angular_thrust += look * (roll_speed * dt);
        }
        if let Some(dt) = self.is_pressed(now, Key::KeyU) {
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
