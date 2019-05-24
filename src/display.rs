use failure::err_msg;
use failure::Error;
use fps_counter::FpsCounter;
use gl;
use gl::types::*;
use input;
use interactive::{DownloadResult, InteractiveKernel};
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
use settings::SettingValueEnum;
use settings::Settings;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct Image {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl Image {
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Image {
        Image {
            data,
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

            texture.update(None, &image.data, image.width as usize * 4)?;
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
        "{} {} {} {} {} {:?}",
        source, type_, id, severity, length, msg
    );
}

#[link(name = "Shell32")]
extern "C" {}

fn check_gl() -> Result<(), Error> {
    loop {
        let er = unsafe { gl::GetError() };
        if er == gl::NO_ERROR {
            return Ok(());
        }
        //println!("OGL error: {}", er);
        return Err(failure::err_msg(format!("OGL error: {}", er)));
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
    settings_input: &Arc<Mutex<(Settings, input::Input)>>,
    time: f32,
) {
    let mut locked = settings_input.lock().unwrap();
    let (ref mut settings, _) = *locked;

    let eye_to_head = system.eye_to_head_transform(eye);
    let head_to_absolute = head.device_to_absolute_tracking();

    let pos = matmul(&head_to_absolute, &matmul(&eye_to_head, &[0.0, 0.0, 0.0]));
    let right = matmul_dir(
        &head_to_absolute,
        &matmul_dir(&eye_to_head, &[1.0, 0.0, 0.0]),
    );
    let up = matmul_dir(
        &head_to_absolute,
        &matmul_dir(&eye_to_head, &[0.0, 1.0, 0.0]),
    );
    let forwards = matmul_dir(
        &head_to_absolute,
        &matmul_dir(&eye_to_head, &[0.0, 0.0, 1.0]),
    );
    //settings.find_mut("PREVIEW").set_const(true);
    *settings.find_mut("pos_x").unwrap_f32_mut() = -pos[0] * 4.0;
    *settings.find_mut("pos_y").unwrap_f32_mut() = -pos[1] * 4.0 + 4.0;
    *settings.find_mut("pos_z").unwrap_f32_mut() = -pos[2] * 4.0;
    *settings.find_mut("look_x").unwrap_f32_mut() = forwards[0];
    *settings.find_mut("look_y").unwrap_f32_mut() = forwards[1];
    *settings.find_mut("look_z").unwrap_f32_mut() = forwards[2];
    *settings.find_mut("up_x").unwrap_f32_mut() = up[0];
    *settings.find_mut("up_y").unwrap_f32_mut() = up[1];
    *settings.find_mut("up_z").unwrap_f32_mut() = up[2];
    // println!(
    //     "({:.2} {:.2} {:.2}) (1:{:.2} {:.2} {:.2}) ({:.2} 1:{:.2} {:.2}) ({:.2} {:.2} :{:.2})",
    //     pos[0],
    //     pos[1],
    //     pos[2],
    //     right[0],
    //     right[1],
    //     right[2],
    //     up[0],
    //     up[1],
    //     up[2],
    //     forwards[0],
    //     forwards[1],
    //     forwards[2],
    // );
}

unsafe fn hands(
    system: &openvr::System,
    compositor: &openvr::Compositor,
    settings_input_left: &Arc<Mutex<(Settings, input::Input)>>,
    settings_input_right: &Arc<Mutex<(Settings, input::Input)>>,
    time: f32,
) -> Result<(), Error> {
    let _ =
        system.tracked_device_index_for_controller_role(openvr::TrackedControllerRole::LeftHand);
    let _ =
        system.tracked_device_index_for_controller_role(openvr::TrackedControllerRole::RightHand);
    let wait_poses: openvr::compositor::WaitPoses = compositor.wait_get_poses()?;

    // render = upcoming frame
    // game = 2 frames from now
    let head: &openvr::TrackedDevicePose = &wait_poses.render[0];

    hands_eye(system, openvr::Eye::Left, head, settings_input_left, time);
    hands_eye(system, openvr::Eye::Right, head, settings_input_right, time);
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

    std::thread::sleep_ms(100);

    {
        let mut locked = settings_input_left.lock().unwrap();
        let (ref mut settings, _) = *locked;
        settings.find_mut("PREVIEW").toggle();
    }

    {
        let mut locked = settings_input_right.lock().unwrap();
        let (ref mut settings, _) = *locked;
        settings.find_mut("PREVIEW").toggle();
    }

    let mut time = 0.0;
    loop {
        time += 1.0 / 90.0;
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
                time,
            )?;
            let left_img = match interactive_kernel_left.download() {
                DownloadResult::Image(img) => Some(img.data),
                DownloadResult::NoMoreImages => return Ok(()),
                DownloadResult::NoneAtPresent => None,
            };
            let right_img = match interactive_kernel_right.download() {
                DownloadResult::Image(img) => Some(img.data),
                DownloadResult::NoMoreImages => return Ok(()),
                DownloadResult::NoneAtPresent => None,
            };
            render_eye(
                &compositor,
                openvr::Eye::Left,
                &mut texture_left,
                left_img.as_ref().map(|x| x as _),
                width as i32,
                height as i32,
            )?;
            render_eye(
                &compositor,
                openvr::Eye::Right,
                &mut texture_right,
                right_img.as_ref().map(|x| x as _),
                width as i32,
                height as i32,
            )?;
        }
    }

    //register_debug()?;

    /*
    let start_image_raw = RawImage2d {
        data: vec![128u8; width as usize * height as usize * 4].into(),
        width: width,
        height: height,
        format: glium::texture::ClientFormat::U8U8U8U8,
    };
    let mut texture = Texture2d::with_format(
        &window,
        start_image_raw,
        glium::texture::UncompressedFloatFormat::U8U8U8U8,
        glium::texture::MipmapsOption::NoMipmap,
    )?;
    // let buffer = Buffer::new(
    //     &window,
    //     &vec![0u8; width as usize * height as usize][..],
    //     glium::buffer::BufferType::ArrayBuffer,
    //     glium::buffer::BufferMode::Persistent,
    // )?;
    //let mut texture: Option<Texture2d> = None;
    // {
    //     let mut surf = texture.as_surface();
    //     surf.clear_all((1.0, 0.5, 1.0, 1.0), 1.0, 0);
    // }

    let settings_input = Arc::new(Mutex::new((Settings::new(), input::Input::new())));
    let interactive_kernel = InteractiveKernel::create(width, height, settings_input.clone())?;

    //let mut fps = FpsCounter::new(1.0);
    let mut running = true;
    let mut event_pump = sdl.event_pump().expect("SDL does not have event pump");
    while running {
        let rect = glium::Rect {
            left: 0,
            bottom: 0,
            width,
            height,
        };
        let mut target = window.draw();
        target.clear_all((1.0, 0.0, 1.0, 1.0), 1.0, 0);
        // draw...

        {
            let brect = glium::BlitTarget {
                left: 0,
                bottom: 0,
                width: (width / 2) as i32,
                height: (height / 2) as i32,
            };
            let tar = texture.as_surface();
            tar.blit_color(
                &rect,
                &target,
                &brect,
                glium::uniforms::MagnifySamplerFilter::Linear,
            );
        }
        target.finish().unwrap();

        // {
        //     let mut surf = texture.as_surface();
        //     surf.clear_all((1.0, 0.5, 1.0, 1.0), 1.0, 0);
        // }

        for event in event_pump.poll_iter() {
            use sdl2::event::Event;
            match event {
                Event::Quit { .. } => {
                    running = false;
                }
                _ => (),
            }
        }

        match interactive_kernel.download() {
            DownloadResult::NoMoreImages => return Ok(()),
            DownloadResult::NoneAtPresent => (),
            DownloadResult::Image(image) => {
                //texture.update(None, &image.data, image.width as usize * 4)?;
                //buffer.write(&image.data);
                //buffer.copy_to(texture);
                let image_raw = RawImage2d {
                    data: image.data.into(),
                    width: width,
                    height: height,
                    format: glium::texture::ClientFormat::U8U8U8U8,
                };
                texture = Texture2d::with_format(
                    &window,
                    image_raw,
                    glium::texture::UncompressedFloatFormat::U8U8U8U8,
                    glium::texture::MipmapsOption::NoMipmap,
                )?;
                //texture.write(rect, image_raw);
                println!("frame");
            }
        }

        {
            let mut locked = settings_input.lock().unwrap();
            let (ref mut settings, _) = *locked;
            if let SettingValueEnum::F32(x, y) = settings.find("pos_x").value {
                settings.find_mut("pos_x").value = SettingValueEnum::F32(x + 0.01, y);
            }
        }

        let _ = compositor.wait_get_poses()?;

        let ovr_tex = openvr::compositor::Texture {
            handle: openvr::compositor::texture::Handle::OpenGLTexture(texture.get_id() as usize),
            color_space: openvr::compositor::texture::ColorSpace::Gamma,
        };
        unsafe {
            match compositor.submit(openvr::Eye::Left, &ovr_tex, None, None) {
                Ok(()) => (),
                Err(err) => println!("Left eye: {}", err),
            }
            match compositor.submit(openvr::Eye::Right, &ovr_tex, None, None) {
                Ok(()) => (),
                Err(err) => println!("Right eye: {}", err),
            }
        }
        //fps.tick();
        //println!("fps: {}", fps.value());
    }
    Ok(())
    // unsafe {
    //     texture.gl_bind_texture();
    //     texture.gl_unbind_texture();
    // }
    //texture.gl_with_bind(|_, _| {
    // let gl_tex_id = get_tex_id();
    // let ovr_tex = openvr::compositor::Texture {
    //     //handle: openvr::compositor::texture::Handle::OpenGLRenderBuffer(),
    //     handle: openvr::compositor::texture::Handle::OpenGLTexture(gl_tex_id),
    //     color_space: openvr::compositor::texture::ColorSpace::Gamma,
    // };
    // unsafe {
    //     compositor
    //         .submit(openvr::Eye::Left, &ovr_tex, None, None)
    //         .expect("Left eye failed to submit");
    //     compositor
    //         .submit(openvr::Eye::Right, &ovr_tex, None, None)
    //         .expect("Right eye failed to submit");
    // }
    //});
    //let _ = compositor.wait_get_poses()?;
    */
}
