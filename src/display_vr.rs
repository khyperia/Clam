use crate::{
    check_gl, display, display::Display, fps_counter::FpsCounter,
    interactive::SyncInteractiveKernel, settings::Settings, Error, Key,
};
use cgmath::{prelude::*, Matrix4, Vector3};
use gl::types::*;
use khygl::{render_text::TextRenderer, render_texture::TextureRenderer, Rect};

fn to_cgmath(mat: [[f32; 4]; 3]) -> Matrix4<f64> {
    Matrix4::new(
        mat[0][0] as f64,
        mat[1][0] as f64,
        mat[2][0] as f64,
        0.0,
        mat[0][1] as f64,
        mat[1][1] as f64,
        mat[2][1] as f64,
        0.0,
        mat[0][2] as f64,
        mat[1][2] as f64,
        mat[2][2] as f64,
        0.0,
        mat[0][3] as f64,
        mat[1][3] as f64,
        mat[2][3] as f64,
        1.0,
    )
}

unsafe fn hands_eye(
    system: &openvr::System,
    eye: openvr::Eye,
    head: &openvr::TrackedDevicePose,
    settings: &mut Settings,
    world: &Matrix4<f64>,
) {
    let eye_to_head = to_cgmath(system.eye_to_head_transform(eye));
    let head_to_absolute = to_cgmath(*head.device_to_absolute_tracking());

    if world[0][0].is_nan() {
        panic!("NaN");
    }

    let world = world
        .inverse_transform()
        .expect("Failed to invert world matrix");

    let pos = world * head_to_absolute * eye_to_head * Vector3::zero().extend(1.0);
    let up = (world * head_to_absolute * eye_to_head * Vector3::new(0.0, 1.0, 0.0).extend(0.0))
        .normalize();
    let forwards =
        (world * head_to_absolute * eye_to_head * Vector3::new(0.0, 0.0, -1.0).extend(0.0))
            .normalize();

    *settings.find_mut("pos").unwrap_vec3_mut() = Vector3::new(pos[0], pos[1], pos[2]) * 8.0;
    *settings.find_mut("look").unwrap_vec3_mut() =
        Vector3::new(forwards[0], forwards[1], forwards[2]);
    *settings.find_mut("up").unwrap_vec3_mut() = Vector3::new(up[0], up[1], up[2]) * 8.0;
}

struct HandsState {
    world: Matrix4<f64>,
    left: Vector3<f64>,
    right: Vector3<f64>,
}

impl HandsState {
    fn new() -> Self {
        Self {
            left: Vector3::new(-1.0, 0.0, 0.0),
            right: Vector3::new(1.0, 0.0, 0.0),
            world: Matrix4::identity(),
        }
    }
}

unsafe fn hands(
    system: &openvr::System,
    compositor: &openvr::Compositor,
    settings_left: &mut Settings,
    settings_right: &mut Settings,
    hands_state: &mut HandsState,
) -> Result<(), Error> {
    let left =
        system.tracked_device_index_for_controller_role(openvr::TrackedControllerRole::LeftHand);
    let right =
        system.tracked_device_index_for_controller_role(openvr::TrackedControllerRole::RightHand);
    let wait_poses: openvr::compositor::WaitPoses = compositor.wait_get_poses()?;

    if let (Some(left), Some(right)) = (left, right) {
        let left_state = system.controller_state(left);
        let right_state = system.controller_state(right);

        let left = &wait_poses.render[left as usize];
        let right = &wait_poses.render[right as usize];
        let left_mat = to_cgmath(*left.device_to_absolute_tracking());
        let right_mat = to_cgmath(*right.device_to_absolute_tracking());
        let left_pos = (left_mat * Vector3::zero().extend(1.0)).truncate();
        let right_pos = (right_mat * Vector3::zero().extend(1.0)).truncate();

        if let (Some(left_state), Some(right_state)) = (left_state, right_state) {
            if left_state.axis[1].x >= 1.0 || right_state.axis[1].x >= 1.0 {
                // let left_delta = sub(&left_pos, &hands_state.left);
                // let right_delta = sub(&right_pos, &hands_state.right);
                // let rotation = mat_scale(1.0); // TODO
                let new_dist = left_pos.distance(right_pos);
                let old_dist = hands_state.left.distance(hands_state.right);
                let scale = new_dist / old_dist;

                let old_center = (hands_state.left + hands_state.right) * 0.5;
                let new_center = (left_pos + right_pos) * 0.5;
                let translation = new_center - old_center;

                fn scale_around(scale: f64, center: Vector3<f64>) -> Matrix4<f64> {
                    Matrix4::from_translation(center)
                        * Matrix4::from_scale(scale)
                        * Matrix4::from_translation(-center)
                }

                //let go_rot = ();
                let go_scale = scale_around(scale, new_center);
                let go_trans = Matrix4::from_translation(translation);
                let transform = go_scale * go_trans;

                println!("{:?}", transform);

                hands_state.world = transform * hands_state.world;
            }
        }

        hands_state.left = left_pos;
        hands_state.right = right_pos;
    }

    // render = upcoming frame
    // game = 2 frames from now
    let head: &openvr::TrackedDevicePose = &wait_poses.render[0];

    hands_eye(
        system,
        openvr::Eye::Left,
        head,
        settings_left,
        &hands_state.world,
    );
    hands_eye(
        system,
        openvr::Eye::Right,
        head,
        settings_right,
        &hands_state.world,
    );
    Ok(())
}

unsafe fn render_eye(
    compositor: &openvr::Compositor,
    eye: openvr::Eye,
    texture: GLuint,
) -> Result<(), Error> {
    check_gl()?;
    let ovr_tex = openvr::compositor::Texture {
        handle: openvr::compositor::texture::Handle::OpenGLTexture(texture as usize),
        color_space: openvr::compositor::texture::ColorSpace::Gamma,
    };
    compositor.submit(eye, &ovr_tex, None, None)?;
    check_gl()?;
    Ok(())
}

struct VrDisplay {
    system: openvr::System,
    compositor: openvr::Compositor,
    interactive_kernel_left: SyncInteractiveKernel<[u8; 4]>,
    interactive_kernel_right: SyncInteractiveKernel<[u8; 4]>,
    hands_state: HandsState,
    texture_renderer: TextureRenderer,
    text_renderer: TextRenderer,
    fps: FpsCounter,
    vr_width: u32,
    vr_height: u32,
}

impl Display for VrDisplay {
    fn setup(_: (usize, usize)) -> Result<Self, Error> {
        let ovr = unsafe { openvr::init(openvr::ApplicationType::Scene) }?;
        let system = ovr.system()?;
        let compositor = ovr.compositor()?;
        let (vr_width, vr_height) = system.recommended_render_target_size();

        let hands_state = HandsState::new();

        let mut interactive_kernel_left =
            SyncInteractiveKernel::<[u8; 4]>::create(vr_width as usize, vr_height as usize)?;
        let mut interactive_kernel_right =
            SyncInteractiveKernel::<[u8; 4]>::create(vr_width as usize, vr_height as usize)?;
        *interactive_kernel_left
            .settings
            .find_mut("VR")
            .unwrap_define_mut() = true;
        *interactive_kernel_right
            .settings
            .find_mut("VR")
            .unwrap_define_mut() = true;
        *interactive_kernel_left
            .settings
            .find_mut("dof_amount")
            .unwrap_float_mut() = 0.0;
        *interactive_kernel_right
            .settings
            .find_mut("dof_amount")
            .unwrap_float_mut() = 0.0;

        let proj_left = system.projection_raw(openvr::Eye::Left);
        *interactive_kernel_left
            .settings
            .find_mut("fov_left")
            .unwrap_float_mut() = proj_left.left as f64;
        *interactive_kernel_left
            .settings
            .find_mut("fov_right")
            .unwrap_float_mut() = proj_left.right as f64;
        *interactive_kernel_left
            .settings
            .find_mut("fov_top")
            .unwrap_float_mut() = proj_left.top as f64;
        *interactive_kernel_left
            .settings
            .find_mut("fov_bottom")
            .unwrap_float_mut() = proj_left.bottom as f64;

        let proj_right = system.projection_raw(openvr::Eye::Right);
        *interactive_kernel_right
            .settings
            .find_mut("fov_left")
            .unwrap_float_mut() = proj_right.left as f64;
        *interactive_kernel_right
            .settings
            .find_mut("fov_right")
            .unwrap_float_mut() = proj_right.right as f64;
        *interactive_kernel_right
            .settings
            .find_mut("fov_top")
            .unwrap_float_mut() = proj_right.top as f64;
        *interactive_kernel_right
            .settings
            .find_mut("fov_bottom")
            .unwrap_float_mut() = proj_right.bottom as f64;

        let fps = FpsCounter::new(1.0);
        let texture_renderer = TextureRenderer::new()?;
        let text_renderer = TextRenderer::new(20.0)?;
        //unsafe { gl::Viewport(0, 0, width as i32, height as i32) };
        unsafe { gl::ClearColor(0.0, 0.0, 0.0, 1.0) };
        check_gl()?;
        Ok(Self {
            system,
            compositor,
            interactive_kernel_left,
            interactive_kernel_right,
            hands_state,
            texture_renderer,
            text_renderer,
            fps,
            vr_width,
            vr_height,
        })
    }

    fn render(&mut self) -> Result<(), Error> {
        unsafe {
            hands(
                &self.system,
                &self.compositor,
                &mut self.interactive_kernel_left.settings,
                &mut self.interactive_kernel_right.settings,
                &mut self.hands_state,
            )?;
        }

        self.interactive_kernel_left.launch()?;
        self.interactive_kernel_right.launch()?;
        let left_img = self.interactive_kernel_left.texture();
        let right_img = self.interactive_kernel_right.texture();

        unsafe {
            render_eye(&self.compositor, openvr::Eye::Left, left_img.id)?;
            render_eye(&self.compositor, openvr::Eye::Right, right_img.id)?;
        }

        unsafe { gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT) };

        self.texture_renderer
            .render(&left_img, (1.0, 1.0))
            .src(Rect::new(0.0, 0.0, 0.5, 1.0))
            .go()?;
        self.texture_renderer
            .render(&right_img, (1.0, 1.0))
            .src(Rect::new(0.5, 0.0, 0.5, 1.0))
            .go()?;

        self.fps.tick();
        let display = format!(
            "{:.2} fps\n{}",
            self.fps.value(),
            self.interactive_kernel_left.status()
        );
        self.text_renderer.render(
            &self.texture_renderer,
            &display,
            [1.0, 0.75, 0.75, 1.0],
            (10, 10),
            (self.vr_width as usize, self.vr_height as usize),
        )?;

        check_gl()?;
        Ok(())
    }

    fn resize(&mut self, _: (usize, usize)) -> Result<(), Error> {
        Ok(())
    }
    fn key_up(&mut self, _: Key) -> Result<(), Error> {
        Ok(())
    }
    fn key_down(&mut self, _: Key) -> Result<(), Error> {
        Ok(())
    }
    fn received_character(&mut self, _: char) -> Result<(), Error> {
        Ok(())
    }
}

pub fn run() -> Result<(), Error> {
    display::run::<VrDisplay>((100.0, 100.0))
}
