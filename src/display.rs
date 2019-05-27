use check_gl;
use failure::Error;
use fps_counter::FpsCounter;
use gl;
use gl::types::*;
use gl_register_debug;
use interactive::SyncInteractiveKernel;
use settings::Settings;
//use openvr;
use sdl2::event::WindowEvent;
use sdl2::init;

pub struct Image {
    pub data_cpu: Option<Vec<f32>>,
    pub data_gl: Option<GLuint>,
    pub width: u32,
    pub height: u32,
}

impl Image {
    pub fn new(
        data_cpu: Option<Vec<f32>>,
        data_gl: Option<GLuint>,
        width: u32,
        height: u32,
    ) -> Image {
        Image {
            data_cpu,
            data_gl,
            width,
            height,
        }
    }
}

unsafe fn buffer_blit(
    buffer: GLuint,
    texture: &mut GLuint,
    framebuffer: &mut GLuint,
    width: i32,
    height: i32,
) -> Result<(), Error> {
    if *texture == 0 {
        let () = gl::CreateTextures(gl::TEXTURE_2D, 1, texture);
        check_gl()?;
        gl::TextureStorage2D(*texture, 1, gl::RGBA32F, width, height);
        check_gl()?;
    }
    if *framebuffer == 0 {
        let () = gl::CreateFramebuffers(1, framebuffer);
        check_gl()?;
        gl::NamedFramebufferTexture(*framebuffer, gl::COLOR_ATTACHMENT0, *texture, 0);
        check_gl()?;
    }

    gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, buffer);
    check_gl()?;
    gl::TextureSubImage2D(
        *texture,
        0,
        0,
        0,
        width,
        height,
        gl::RGBA,
        gl::FLOAT,
        std::ptr::null(),
    );
    check_gl()?;
    gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
    check_gl()?;

    if *framebuffer != 0 {
        let dest_buf = 0;
        gl::BlitNamedFramebuffer(
            *framebuffer,
            dest_buf,
            0,
            0,
            width,
            height,
            0,
            0,
            width,
            height,
            gl::COLOR_BUFFER_BIT,
            gl::NEAREST,
        );
        check_gl()?;
    }
    Ok(())
}

pub fn gl_display(init_width: u32, init_height: u32) -> Result<(), Error> {
    let is_gl = true;
    let sdl = init().expect("SDL failed to init");
    let video = sdl.video().expect("SDL does not have video");
    let mut event_pump = sdl.event_pump().expect("SDL does not have event pump");

    video.gl_attr().set_context_flags().debug().set();

    let window = video
        .window("clam5", init_width, init_height)
        .opengl()
        .build()?;
    let _gl_context = window
        .gl_create_context()
        .expect("Failed to create OpenGL context");

    gl::load_with(|s| video.gl_get_proc_address(s) as *const _);

    if !gl::GetError::is_loaded() {
        return Err(failure::err_msg("glGetError not loaded"));
    }

    unsafe { gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS) };
    check_gl()?;

    gl_register_debug()?;

    let mut interactive_kernel =
        SyncInteractiveKernel::create(init_width, init_height, is_gl, Some(&video))?;

    let mut fps = FpsCounter::new(1.0);

    let mut texture = 0;
    let mut framebuffer = 0;
    loop {
        for event in event_pump.poll_iter() {
            use sdl2::event::Event;
            match event {
                Event::Window {
                    win_event: WindowEvent::Resized(width, height),
                    ..
                } if width > 0 && height > 0 => {
                    interactive_kernel.resize(width as u32, height as u32)?;
                }
                Event::KeyDown {
                    scancode: Some(scancode),
                    ..
                } => {
                    interactive_kernel.key_down(scancode);
                }
                Event::KeyUp {
                    scancode: Some(scancode),
                    ..
                } => {
                    interactive_kernel.key_up(scancode);
                }
                Event::Quit { .. } => return Ok(()),
                _ => (),
            }
        }

        interactive_kernel.launch()?;
        let img = interactive_kernel.download()?;

        unsafe {
            buffer_blit(
                img.data_gl.expect("gl_display needs OGL texture"),
                &mut texture,
                &mut framebuffer,
                img.width as _,
                img.height as _,
            )
        }?;

        interactive_kernel.print_status(&fps);

        window.gl_swap_window();

        fps.tick();
    }
}

fn matmul(mat: &[[f32; 4]; 3], vec: &[f32; 3]) -> [f32; 3] {
    [
        mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2] + mat[0][3],
        mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2] + mat[1][3],
        mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2] + mat[2][3],
    ]
}

fn matmul_dir(mat: &[[f32; 4]; 3], vec: &[f32; 3]) -> [f32; 3] {
    [
        mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2],
        mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2],
        mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2],
    ]
}

unsafe fn hands_eye(
    system: &openvr::System,
    eye: openvr::Eye,
    head: &openvr::TrackedDevicePose,
    settings: &mut Settings,
) {
    let eye_to_head = system.eye_to_head_transform(eye);
    let head_to_absolute = head.device_to_absolute_tracking();

    let pos = matmul(&head_to_absolute, &matmul(&eye_to_head, &[0.0, 0.0, 0.0]));
    // let right = matmul_dir(
    //     &head_to_absolute,
    //     &matmul_dir(&eye_to_head, &[1.0, 0.0, 0.0]),
    // );
    let up = matmul_dir(
        &head_to_absolute,
        &matmul_dir(&eye_to_head, &[0.0, 1.0, 0.0]),
    );
    let forwards = matmul_dir(
        &head_to_absolute,
        &matmul_dir(&eye_to_head, &[0.0, 0.0, 1.0]),
    );
    settings.find_mut("VR").set_const(true);
    *settings.find_mut("pos_x").unwrap_f32_mut() = -pos[0] * 4.0;
    *settings.find_mut("pos_y").unwrap_f32_mut() = -pos[1] * 4.0 + 4.0;
    *settings.find_mut("pos_z").unwrap_f32_mut() = -pos[2] * 4.0;
    *settings.find_mut("look_x").unwrap_f32_mut() = forwards[0];
    *settings.find_mut("look_y").unwrap_f32_mut() = forwards[1];
    *settings.find_mut("look_z").unwrap_f32_mut() = forwards[2];
    *settings.find_mut("up_x").unwrap_f32_mut() = up[0];
    *settings.find_mut("up_y").unwrap_f32_mut() = up[1];
    *settings.find_mut("up_z").unwrap_f32_mut() = up[2];
}

unsafe fn hands(
    system: &openvr::System,
    compositor: &openvr::Compositor,
    settings_left: &mut Settings,
    settings_right: &mut Settings,
) -> Result<(), Error> {
    let _ =
        system.tracked_device_index_for_controller_role(openvr::TrackedControllerRole::LeftHand);
    let _ =
        system.tracked_device_index_for_controller_role(openvr::TrackedControllerRole::RightHand);
    let wait_poses: openvr::compositor::WaitPoses = compositor.wait_get_poses()?;

    // render = upcoming frame
    // game = 2 frames from now
    let head: &openvr::TrackedDevicePose = &wait_poses.render[0];

    hands_eye(system, openvr::Eye::Left, head, settings_left);
    hands_eye(system, openvr::Eye::Right, head, settings_right);
    Ok(())
}

unsafe fn render_eye(
    compositor: &openvr::Compositor,
    eye: openvr::Eye,
    buffer: GLuint,
    texture: &mut GLuint,
    width: GLsizei,
    height: GLsizei,
) -> Result<(), Error> {
    check_gl()?;
    if *texture == 0 {
        let () = gl::CreateTextures(gl::TEXTURE_2D, 1, texture);
        check_gl()?;
        //gl::TextureBuffer(texture, gl::RGBA8UI, buffer);
        gl::TextureStorage2D(*texture, 1, gl::RGBA8UI, width, height);
        check_gl()?;
    }

    gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, buffer);
    check_gl()?;
    gl::TextureSubImage2D(
        *texture,
        0,
        0,
        0,
        width,
        height,
        gl::RGBA_INTEGER,
        //gl::FLOAT,
        gl::UNSIGNED_INT_8_8_8_8,
        std::ptr::null(),
    );
    check_gl()?;
    gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
    check_gl()?;

    let ovr_tex = openvr::compositor::Texture {
        //handle: openvr::compositor::texture::Handle::OpenGLRenderBuffer(),
        handle: openvr::compositor::texture::Handle::OpenGLTexture(*texture as usize),
        color_space: openvr::compositor::texture::ColorSpace::Gamma,
    };
    compositor
        .submit(eye, &ovr_tex, None, None)
        .expect("Eye failed to submit");
    check_gl()?;
    Ok(())
}

pub fn vr_display() -> Result<(), Error> {
    let is_gl = true;
    let sdl = init().expect("SDL failed to init");
    let video = sdl.video().expect("SDL does not have video");
    let mut event_pump = sdl.event_pump().expect("SDL does not have event pump");

    video.gl_attr().set_context_flags().debug().set();

    let window = video.window("clam5", 500, 500).opengl().build()?;
    let _gl_context = window
        .gl_create_context()
        .expect("Failed to create OpenGL context");

    gl::load_with(|s| video.gl_get_proc_address(s) as *const _);

    if !gl::GetError::is_loaded() {
        return Err(failure::err_msg("glGetError not loaded"));
    }

    unsafe { gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS) };
    check_gl()?;

    gl_register_debug()?;

    let ovr = unsafe { openvr::init(openvr::ApplicationType::Scene)? };
    let system = ovr.system()?;
    let compositor = ovr.compositor()?;
    let (width, height) = system.recommended_render_target_size();
    check_gl()?;

    let mut interactive_kernel_left =
        SyncInteractiveKernel::create(width, height, is_gl, Some(&video))?;
    let mut interactive_kernel_right =
        SyncInteractiveKernel::create(width, height, is_gl, Some(&video))?;

    let mut fps = FpsCounter::new(1.0);

    let mut left_texture = 0;
    let mut right_texture = 0;
    loop {
        for event in event_pump.poll_iter() {
            use sdl2::event::Event;
            match event {
                Event::Quit { .. } => return Ok(()),
                _ => (),
            }
        }

        unsafe {
            hands(
                &system,
                &compositor,
                &mut interactive_kernel_left.settings,
                &mut interactive_kernel_right.settings,
            )?;
        }

        interactive_kernel_left.launch()?;
        interactive_kernel_right.launch()?;
        let left_img = interactive_kernel_left.download()?;
        let right_img = interactive_kernel_right.download()?;

        unsafe {
            render_eye(
                &compositor,
                openvr::Eye::Left,
                left_img.data_gl.expect("vr_display needs OGL textures"),
                &mut left_texture,
                width as i32,
                height as i32,
            )?;
            render_eye(
                &compositor,
                openvr::Eye::Right,
                right_img.data_gl.expect("vr_display needs OGL textures"),
                &mut right_texture,
                width as i32,
                height as i32,
            )?;
        }

        window.gl_swap_window();

        fps.tick();
        println!("{:.2}fps", fps.value());
    }
}
