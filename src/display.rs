use check_gl;
use failure::err_msg;
use failure::Error;
use fps_counter::FpsCounter;
use gl;
use gl::types::*;
use input;
use interactive::{DownloadResult, InteractiveKernel, SyncInteractiveKernel};
use openvr;
use sdl2::event::Event;
use sdl2::event::WindowEvent;
use sdl2::init;
use sdl2::pixels::Color;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use sdl2::render::Canvas;
use sdl2::render::Texture;
use sdl2::render::TextureCreator;
use sdl2::ttf;
use sdl2::ttf::Font;
use sdl2::video::Window;
use sdl2::video::WindowContext;
use settings::Settings;
use std::path::Path;
use std::sync::{Arc, Mutex};

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

pub enum ScreenEvent {
    Resize(u32, u32),
    KernelChanged,
}

fn find_font() -> Result<&'static Path, Error> {
    let locations: [&'static Path; 6] = [
        "/usr/share/fonts/TTF/FiraMono-Regular.ttf".as_ref(),
        "/usr/share/fonts/TTF/FiraSans-Regular.ttf".as_ref(),
        "C:\\Windows\\Fonts\\arial.ttf".as_ref(),
        "/usr/share/fonts/TTF/DejaVuSans.ttf".as_ref(),
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf".as_ref(),
        "/Library/Fonts/Andale Mono.ttf".as_ref(),
    ];
    for &location in &locations {
        if location.exists() {
            return Ok(location);
        }
    }
    Err(err_msg("No font found"))
}

fn render_text_one(
    font: &ttf::Font,
    creator: &TextureCreator<WindowContext>,
    canvas: &mut Canvas<Window>,
    text: &str,
    color: Color,
    offset_x: i32,
    offset_y: i32,
) -> Result<(), Error> {
    let (_, window_height) = canvas.output_size().map_err(err_msg)?;
    let spacing = font.recommended_line_spacing();

    let mut current_y = 10;
    let mut current_column_x = 10;
    let mut next_column_x = 0;

    for line in text.lines() {
        let rendered = font.render(line).solid(color)?;
        let width = rendered.width();
        let height = rendered.height();
        let tex = creator.create_texture_from_surface(rendered)?;
        if (current_y + spacing) >= (window_height as i32) {
            current_column_x = next_column_x;
            current_y = 10;
        }
        next_column_x = next_column_x.max(current_column_x + width as i32);
        canvas
            .copy(
                &tex,
                None,
                Rect::new(
                    current_column_x + offset_x,
                    current_y + offset_y,
                    width,
                    height,
                ),
            )
            .expect("Could not display text");
        current_y += spacing;
    }
    Ok(())
}

fn render_text(
    font: &ttf::Font,
    settings_input: &Arc<Mutex<(Settings, input::Input)>>,
    creator: &TextureCreator<WindowContext>,
    canvas: &mut Canvas<Window>,
    render_fps: f64,
    window_fps: f64,
) -> Result<(), Error> {
    let fps_line_1 = format!("{:.2} render fps", render_fps);
    let fps_line_2 = format!("{:.2} window fps", window_fps);
    let mut locked = settings_input.lock().unwrap();
    let (ref mut settings, ref mut input) = *locked;
    input.integrate(settings);
    let text = format!("{}\n{}\n{}", fps_line_1, fps_line_2, settings.status(input));
    render_text_one(font, creator, canvas, &text, Color::RGB(0, 0, 0), 1, 1)?;
    render_text_one(
        font,
        creator,
        canvas,
        &text,
        Color::RGB(255, 192, 192),
        0,
        0,
    )?;
    Ok(())
}

struct WindowData {
    image_width: u32,
    image_height: u32,
    render_fps: FpsCounter,
    window_fps: FpsCounter,
}

fn draw<'a>(
    canvas: &mut Canvas<Window>,
    creator: &'a TextureCreator<WindowContext>,
    font: &Font,
    interactive_kernel: &InteractiveKernel,
    settings_input: &Arc<Mutex<(Settings, input::Input)>>,
    texture: &mut Texture<'a>,
    window_data: &mut WindowData,
) -> Result<bool, Error> {
    match interactive_kernel.download() {
        DownloadResult::NoMoreImages => return Ok(false),
        DownloadResult::NoneAtPresent => (),
        DownloadResult::Image(image) => {
            window_data.render_fps.tick();
            if window_data.image_width != image.width || window_data.image_height != image.height {
                window_data.image_width = image.width;
                window_data.image_height = image.height;
                *texture = creator.create_texture_streaming(
                    PixelFormatEnum::ABGR8888,
                    window_data.image_width,
                    window_data.image_height,
                )?;
            }

            let image_ = image
                .data_cpu
                .expect("draw() expects CPU image")
                .into_iter()
                .map(::f32_to_u8)
                .collect::<Vec<_>>();

            texture.update(None, &image_, image.width as usize * 4)?;
        }
    }
    // let image = match image_stream.try_recv() {
    //     Ok(image) => Some(image),
    //     Err(mpsc::TryRecvError::Empty) => None,
    //     Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
    // };

    //if let Some(image) = image {
    //}

    window_data.window_fps.tick();

    let (output_width, output_height) = canvas.output_size().map_err(failure::err_msg)?;
    let src = Rect::new(0, 0, window_data.image_width, window_data.image_height);
    let dest = Rect::new(0, 0, output_width, output_height);
    canvas
        .copy(texture, src, dest)
        .expect("Could not display image");
    render_text(
        font,
        settings_input,
        creator,
        canvas,
        window_data.render_fps.value(),
        window_data.window_fps.value(),
    )?;
    canvas.present();
    Ok(true)
}

pub fn display(width: u32, height: u32) -> Result<(), Error> {
    let sdl = init().expect("SDL failed to init");
    let video = sdl.video().expect("SDL does not have video");
    let mut event_pump = sdl.event_pump().expect("SDL does not have event pump");

    let window = video.window("clam5", width, height).resizable().build()?;
    let mut canvas = window.into_canvas().present_vsync().build()?;
    let creator = canvas.texture_creator();
    let mut texture = creator.create_texture_streaming(PixelFormatEnum::ABGR8888, width, height)?;

    let ttf = ttf::init()?;
    let font = ttf.load_font(find_font()?, 20).expect("Cannot open font");

    let settings_input = Arc::new(Mutex::new((Settings::new(), input::Input::new())));

    let interactive_kernel = InteractiveKernel::create(width, height, settings_input.clone())?;

    let mut window_data = WindowData {
        image_width: width,
        image_height: height,
        render_fps: FpsCounter::new(1.0),
        window_fps: FpsCounter::new(1.0),
    };

    loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Window {
                    win_event: WindowEvent::Resized(width, height),
                    ..
                } if width > 0 && height > 0 => {
                    interactive_kernel.resize(width as u32, height as u32);
                    // match event_stream.send(ScreenEvent::Resize(width as u32, height as u32)) {
                    //     Ok(()) => (),
                    //     Err(_) => return Ok(()),
                    // }
                }
                Event::KeyDown {
                    scancode: Some(scancode),
                    ..
                } => {
                    let mut locked = settings_input.lock().unwrap();
                    let (ref mut settings, ref mut input) = *locked;
                    input.key_down(scancode, settings);
                }
                Event::KeyUp {
                    scancode: Some(scancode),
                    ..
                } => {
                    let mut locked = settings_input.lock().unwrap();
                    let (ref mut settings, ref mut input) = *locked;
                    input.key_up(scancode, settings);
                }
                Event::Quit { .. } => return Ok(()),
                _ => (),
            }
        }
        if !draw(
            &mut canvas,
            &creator,
            &font,
            &interactive_kernel,
            &settings_input,
            &mut texture,
            &mut window_data,
        )? {
            return Ok(());
        }
    }
}

unsafe fn buffer_blit(
    buffer: Option<GLuint>,
    texture: &mut GLuint,
    framebuffer: &mut GLuint,
    width: i32,
    height: i32,
) -> Result<(), Error> {
    if *texture == 0 {
        let () = gl::CreateTextures(gl::TEXTURE_2D, 1, texture);
        ::check_gl()?;
        gl::TextureStorage2D(*texture, 1, gl::RGBA32F, width, height);
        ::check_gl()?;
    }
    if *framebuffer == 0 {
        let () = gl::CreateFramebuffers(1, framebuffer);
        ::check_gl()?;
        gl::NamedFramebufferTexture(*framebuffer, gl::COLOR_ATTACHMENT0, *texture, 0);
        ::check_gl()?;
    }
    if let Some(buffer) = buffer {
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, buffer);
        ::check_gl()?;
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
        ::check_gl()?;
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
        ::check_gl()?;
    }

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
        ::check_gl()?;
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

    println!(
        "wglGetCurrentContext: {:?}",
        video.gl_get_proc_address("wglGetCurrentContext")
    );

    if !gl::GetError::is_loaded() {
        println!("GetError not loaded");
        return Ok(());
    }

    check_gl()?;
    if !gl::Viewport::is_loaded() {
        println!("Viewport not loaded");
        return Ok(());
    }

    unsafe { gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS) };
    check_gl()?;

    register_debug()?;
    check_gl()?;

    let settings_input = Arc::new(Mutex::new((Settings::new(), input::Input::new())));
    let mut interactive_kernel = SyncInteractiveKernel::create(
        init_width,
        init_height,
        is_gl,
        Some((&window, &_gl_context)),
        settings_input.clone(),
    )?;

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
                    let mut locked = settings_input.lock().unwrap();
                    let (ref mut settings, ref mut input) = *locked;
                    input.key_down(scancode, settings);
                }
                Event::KeyUp {
                    scancode: Some(scancode),
                    ..
                } => {
                    let mut locked = settings_input.lock().unwrap();
                    let (ref mut settings, ref mut input) = *locked;
                    input.key_up(scancode, settings);
                }
                Event::Quit { .. } => return Ok(()),
                _ => (),
            }
        }

        let img = interactive_kernel.download()?;

        unsafe {
            buffer_blit(
                img.data_gl,
                &mut texture,
                &mut framebuffer,
                img.width as _,
                img.height as _,
            )
        }?;

        window.gl_swap_window();
    }
}

fn register_debug() -> Result<(), Error> {
    unsafe {
        gl::DebugMessageCallback(debug_callback, std::ptr::null());
    }
    Ok(())
}

extern "system" fn debug_callback(
    source: u32,
    type_: u32,
    id: u32,
    severity: u32,
    length: i32,
    message: *const i8,
    _: *mut libc::c_void,
) {
    let msg = std::str::from_utf8(unsafe {
        std::slice::from_raw_parts(message as *const u8, length as usize)
    });
    println!(
        "GL debug callback: {} {} {} {} {} {:?}",
        source, type_, id, severity, length, msg
    );
}

#[link(name = "Shell32")]
extern "C" {}

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
    settings_input: &Arc<Mutex<(Settings, input::Input)>>,
) {
    let mut locked = settings_input.lock().unwrap();
    let (ref mut settings, _) = *locked;

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
    settings_input_left: &Arc<Mutex<(Settings, input::Input)>>,
    settings_input_right: &Arc<Mutex<(Settings, input::Input)>>,
) -> Result<(), Error> {
    let _ =
        system.tracked_device_index_for_controller_role(openvr::TrackedControllerRole::LeftHand);
    let _ =
        system.tracked_device_index_for_controller_role(openvr::TrackedControllerRole::RightHand);
    let wait_poses: openvr::compositor::WaitPoses = compositor.wait_get_poses()?;

    // render = upcoming frame
    // game = 2 frames from now
    let head: &openvr::TrackedDevicePose = &wait_poses.render[0];

    hands_eye(system, openvr::Eye::Left, head, settings_input_left);
    hands_eye(system, openvr::Eye::Right, head, settings_input_right);
    Ok(())
}

unsafe fn render_eye(
    compositor: &openvr::Compositor,
    eye: openvr::Eye,
    texture: &mut GLuint,
    data: Option<&[u8]>,
    width: i32,
    height: i32,
) -> Result<(), Error> {
    check_gl()?;
    if let Some(data) = data {
        if *texture == 0 {
            let () = gl::GenTextures(1, texture);
            check_gl()?;
            gl::BindTexture(gl::TEXTURE_2D, *texture);
            check_gl()?;
            gl::TexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA8I, width, height);
            check_gl()?;
        } else {
            gl::BindTexture(gl::TEXTURE_2D, *texture);
        }
        gl::TexSubImage2D(
            gl::TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            gl::RGBA_INTEGER,
            gl::UNSIGNED_INT_8_8_8_8,
            data.as_ptr() as _,
        );
        check_gl()?;
    }
    if *texture != 0 {
        let ovr_tex = openvr::compositor::Texture {
            //handle: openvr::compositor::texture::Handle::OpenGLRenderBuffer(),
            handle: openvr::compositor::texture::Handle::OpenGLTexture(*texture as usize),
            color_space: openvr::compositor::texture::ColorSpace::Gamma,
        };
        compositor
            .submit(eye, &ovr_tex, None, None)
            .expect("Eye failed to submit");
        check_gl()?;
    }
    Ok(())
}

pub fn vr_display() -> Result<(), Error> {
    let sdl = init().expect("SDL failed to init");
    let video = sdl.video().expect("SDL does not have video");
    let mut event_pump = sdl.event_pump().expect("SDL does not have event pump");
    let window = video
        //.window("clam5", width / 2, height / 2)
        .window("clam5", 500, 500)
        .opengl()
        .build()?;
    let _gl_context = window
        .gl_create_context()
        .expect("Failed to create OpenGL context");

    gl::load_with(|s| video.gl_get_proc_address(s) as *const _);

    if !gl::GetError::is_loaded() {
        println!("GetError not loaded");
        return Ok(());
    }

    check_gl()?;
    if !gl::Viewport::is_loaded() {
        println!("Viewport not loaded");
        return Ok(());
    }

    register_debug()?;
    check_gl()?;

    let ovr = unsafe { openvr::init(openvr::ApplicationType::Scene)? };
    let system = ovr.system()?;
    let compositor = ovr.compositor()?;
    let (width, height) = system.recommended_render_target_size();
    check_gl()?;

    let settings_input_left = Arc::new(Mutex::new((Settings::new(), input::Input::new())));
    let interactive_kernel_left =
        InteractiveKernel::create(width, height, settings_input_left.clone())?;
    let settings_input_right = Arc::new(Mutex::new((Settings::new(), input::Input::new())));
    let interactive_kernel_right =
        InteractiveKernel::create(width, height, settings_input_right.clone())?;

    let mut texture_left = 0;
    let mut texture_right = 0;

    std::thread::sleep(std::time::Duration::from_millis(100));

    loop {
        for event in event_pump.poll_iter() {
            use sdl2::event::Event;
            match event {
                Event::Quit { .. } => {
                    return Ok(());
                }
                _ => (),
            }
        }

        check_gl()?;
        unsafe {
            hands(
                &system,
                &compositor,
                &settings_input_left,
                &settings_input_right,
            )?;
            let left_img = match interactive_kernel_left.download() {
                DownloadResult::Image(img) => {
                    Some(img.data_cpu.expect("vr_display expects CPU image"))
                }
                DownloadResult::NoMoreImages => return Ok(()),
                DownloadResult::NoneAtPresent => None,
            };
            let right_img = match interactive_kernel_right.download() {
                DownloadResult::Image(img) => {
                    Some(img.data_cpu.expect("vr_display expects CPU image"))
                }
                DownloadResult::NoMoreImages => return Ok(()),
                DownloadResult::NoneAtPresent => None,
            };

            let left_image_ = left_img.map(|x| x.into_iter().map(::f32_to_u8).collect::<Vec<_>>());
            let right_image_ = right_img.map(|x| x.into_iter().map(::f32_to_u8).collect::<Vec<_>>());

            render_eye(
                &compositor,
                openvr::Eye::Left,
                &mut texture_left,
                left_image_.as_ref().map(|x| x as _),
                width as i32,
                height as i32,
            )?;
            render_eye(
                &compositor,
                openvr::Eye::Right,
                &mut texture_right,
                right_image_.as_ref().map(|x| x as _),
                width as i32,
                height as i32,
            )?;
        }
    }
}
