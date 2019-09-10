use crate::check_gl;
use crate::fps_counter::FpsCounter;
use crate::gl_register_debug;
use crate::interactive::SyncInteractiveKernel;
use crate::kernel;
use crate::render_text::TextRenderer;
use crate::render_texture::TextureRenderer;
use crate::render_texture::TextureRendererKind;
use crate::settings::Settings;
use cgmath::prelude::*;
use cgmath::Matrix4;
use cgmath::Vector3;
use failure::err_msg;
use failure::Error;
use gl::types::*;
use sdl2::event::Event;
use sdl2::event::WindowEvent;
use sdl2::init;
use sdl2::pixels::Color;

/*
fn mat_print(mat: &[[f32; 4]; 3]) {
    println!(
        "[{}, {}, {}, {}]",
        mat[0][0], mat[0][1], mat[0][2], mat[0][3]
    );
    println!(
        "[{}, {}, {}, {}]",
        mat[1][0], mat[1][1], mat[1][2], mat[1][3]
    );
    println!(
        "[{}, {}, {}, {}]",
        mat[2][0], mat[2][1], mat[2][2], mat[2][3]
    );
}
*/

fn to_cgmath(mat: [[f32; 4]; 3]) -> Matrix4<f32> {
    Matrix4::new(
        mat[0][0], mat[1][0], mat[2][0], 0.0, mat[0][1], mat[1][1], mat[2][1], 0.0, mat[0][2],
        mat[1][2], mat[2][2], 0.0, mat[0][3], mat[1][3], mat[2][3], 1.0,
    )
}

unsafe fn hands_eye(
    system: &openvr::System,
    eye: openvr::Eye,
    head: &openvr::TrackedDevicePose,
    settings: &mut Settings,
    world: &Matrix4<f32>,
) {
    let eye_to_head = to_cgmath(system.eye_to_head_transform(eye));
    let head_to_absolute = to_cgmath(*head.device_to_absolute_tracking());

    if world[0][0].is_nan() {
        panic!("NaN");
    }

    let world = world.inverse_transform().unwrap();

    let pos = world * head_to_absolute * eye_to_head * Vector3::zero().extend(1.0);
    let up = (world * head_to_absolute * eye_to_head * Vector3::new(0.0, 1.0, 0.0).extend(0.0))
        .normalize();
    let forwards =
        (world * head_to_absolute * eye_to_head * Vector3::new(0.0, 0.0, -1.0).extend(0.0))
            .normalize();

    *settings.find_mut("pos_x").unwrap_f32_mut() = pos[0] * 8.0;
    *settings.find_mut("pos_y").unwrap_f32_mut() = pos[1] * 8.0;
    *settings.find_mut("pos_z").unwrap_f32_mut() = pos[2] * 8.0;
    *settings.find_mut("look_x").unwrap_f32_mut() = forwards[0];
    *settings.find_mut("look_y").unwrap_f32_mut() = forwards[1];
    *settings.find_mut("look_z").unwrap_f32_mut() = forwards[2];
    *settings.find_mut("up_x").unwrap_f32_mut() = up[0];
    *settings.find_mut("up_y").unwrap_f32_mut() = up[1];
    *settings.find_mut("up_z").unwrap_f32_mut() = up[2];
}

struct HandsState {
    world: Matrix4<f32>,
    left: Vector3<f32>,
    right: Vector3<f32>,
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

                fn scale_around(scale: f32, center: Vector3<f32>) -> Matrix4<f32> {
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

pub fn vr_display() -> Result<(), Error> {
    let is_gl = true;
    let sdl = init().map_err(err_msg)?;
    let video = sdl.video().map_err(err_msg)?;
    let mut event_pump = sdl.event_pump().map_err(err_msg)?;

    video.gl_attr().set_context_flags().debug().set();

    let ovr = unsafe { openvr::init(openvr::ApplicationType::Scene)? };
    let system = ovr.system()?;
    let compositor = ovr.compositor()?;
    let (width, height) = system.recommended_render_target_size();

    let mut screen_width = width;
    let mut screen_height = height / 2;
    let window = video
        .window("clam5", screen_width, screen_height)
        .resizable()
        .opengl()
        .build()?;
    let _gl_context = window.gl_create_context().map_err(err_msg)?;

    gl::load_with(|s| video.gl_get_proc_address(s) as *const _);

    if !gl::GetError::is_loaded() {
        return Err(failure::err_msg("glGetError not loaded"));
    }

    unsafe { gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS) };
    check_gl()?;

    gl_register_debug()?;

    kernel::init_gl_funcs(&video);

    check_gl()?;

    let mut hands_state = HandsState::new();

    let mut interactive_kernel_left = SyncInteractiveKernel::<u8>::create(width, height, is_gl)?;
    let mut interactive_kernel_right = SyncInteractiveKernel::<u8>::create(width, height, is_gl)?;
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
        .unwrap_f32_mut() = 0.0;
    *interactive_kernel_right
        .settings
        .find_mut("dof_amount")
        .unwrap_f32_mut() = 0.0;

    let proj_left = system.projection_raw(openvr::Eye::Left);
    *interactive_kernel_left
        .settings
        .find_mut("fov_left")
        .unwrap_f32_mut() = proj_left.left;
    *interactive_kernel_left
        .settings
        .find_mut("fov_right")
        .unwrap_f32_mut() = proj_left.right;
    *interactive_kernel_left
        .settings
        .find_mut("fov_top")
        .unwrap_f32_mut() = proj_left.top;
    *interactive_kernel_left
        .settings
        .find_mut("fov_bottom")
        .unwrap_f32_mut() = proj_left.bottom;

    let proj_right = system.projection_raw(openvr::Eye::Right);
    *interactive_kernel_right
        .settings
        .find_mut("fov_left")
        .unwrap_f32_mut() = proj_right.left;
    *interactive_kernel_right
        .settings
        .find_mut("fov_right")
        .unwrap_f32_mut() = proj_right.right;
    *interactive_kernel_right
        .settings
        .find_mut("fov_top")
        .unwrap_f32_mut() = proj_right.top;
    *interactive_kernel_right
        .settings
        .find_mut("fov_bottom")
        .unwrap_f32_mut() = proj_right.bottom;

    interactive_kernel_left.settings.rebuild();
    interactive_kernel_right.settings.rebuild();

    let mut fps = FpsCounter::new(1.0);
    let texture_renderer_u8 = TextureRenderer::new(TextureRendererKind::U8);
    let texture_renderer_f32 = TextureRenderer::new(TextureRendererKind::F32);
    let text_renderer = TextRenderer::new(Color::RGB(255, 192, 192));
    unsafe { gl::Viewport(0, 0, screen_width as i32, screen_height as i32) };
    unsafe { gl::ClearColor(0.0, 0.0, 0.0, 1.0) };
    check_gl()?;

    loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::Window {
                    win_event: WindowEvent::Close,
                    ..
                } => return Ok(()),
                Event::Window {
                    win_event: WindowEvent::Resized(width, height),
                    window_id,
                    ..
                } if window_id == window.id() && width > 0 && height > 0 => {
                    screen_width = width as u32;
                    screen_height = height as u32;
                    unsafe { gl::Viewport(0, 0, width, height) };
                }
                _ => (),
            }
        }

        unsafe {
            hands(
                &system,
                &compositor,
                &mut interactive_kernel_left.settings,
                &mut interactive_kernel_right.settings,
                &mut hands_state,
            )?;
        }

        interactive_kernel_left.launch()?;
        interactive_kernel_right.launch()?;
        let left_img = interactive_kernel_left
            .download()?
            .data_gl
            .expect("vr_display needs OGL textures");
        let right_img = interactive_kernel_right
            .download()?
            .data_gl
            .expect("vr_display needs OGL textures");

        unsafe {
            render_eye(&compositor, openvr::Eye::Left, left_img)?;
            render_eye(&compositor, openvr::Eye::Right, right_img)?;
        }

        unsafe { gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT) };

        texture_renderer_u8.render(left_img, 0.0, 0.0, 0.5, 1.0)?;
        texture_renderer_u8.render(right_img, 0.5, 0.0, 0.5, 1.0)?;

        fps.tick();
        let display = format!(
            "{:.2} fps\n{}",
            fps.value(),
            interactive_kernel_left.status()
        );
        text_renderer.render(&texture_renderer_f32, &display, screen_width, screen_height)?;

        window.gl_swap_window();

        check_gl()?;
    }
}
