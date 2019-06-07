use crate::check_gl;
use crate::fps_counter::FpsCounter;
use crate::gl_register_debug;
use crate::interactive::SyncInteractiveKernel;
use crate::kernel;
use failure::err_msg;
use failure::Error;
use gl;
use gl::types::*;
use sdl2::event::Event;
use sdl2::event::WindowEvent;
use sdl2::init;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::Canvas;
use sdl2::render::TextureCreator;
use sdl2::ttf;
use sdl2::video::Window;
use sdl2::video::WindowContext;
use std::path::Path;

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

fn render_text(
    font: &ttf::Font,
    creator: &TextureCreator<WindowContext>,
    canvas: &mut Canvas<Window>,
    text: &str,
) -> Result<(), Error> {
    let color = Color::RGB(255, 192, 192);
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
                Rect::new(current_column_x, current_y, width, height),
            )
            .map_err(err_msg)?;
        current_y += spacing;
    }
    Ok(())
}

unsafe fn buffer_blit(
    buffer: GLuint,
    framebuffer: &mut GLuint,
    image_width: i32,
    image_height: i32,
    screen_width: i32,
    screen_height: i32,
) -> Result<(), Error> {
    if *framebuffer == 0 {
        gl::CreateFramebuffers(1, framebuffer);
        check_gl()?;
        gl::NamedFramebufferTexture(*framebuffer, gl::COLOR_ATTACHMENT0, buffer, 0);
        check_gl()?;
    }

    let dest_buf = 0;
    gl::BlitNamedFramebuffer(
        *framebuffer,
        dest_buf,
        0,
        0,
        image_width,
        image_height,
        0,
        0,
        screen_width,
        screen_height,
        gl::COLOR_BUFFER_BIT,
        gl::NEAREST,
    );
    check_gl()?;
    Ok(())
}

pub fn gl_display(mut screen_width: u32, mut screen_height: u32) -> Result<(), Error> {
    let is_gl = true;
    let sdl = init().map_err(err_msg)?;
    let video = sdl.video().map_err(err_msg)?;
    let mut event_pump = sdl.event_pump().map_err(err_msg)?;

    let ttf = ttf::init()?;
    let font = ttf.load_font(find_font()?, 20).map_err(err_msg)?;

    video.gl_attr().set_context_flags().debug().set();

    let window = video
        .window("clam5", screen_width, screen_height)
        .opengl()
        .build()?;
    let _gl_context = window.gl_create_context().map_err(err_msg)?;

    let mut window_text = video
        .window("clam5 data", 512, 512)
        .resizable()
        .build()?
        .into_canvas()
        .build()?;
    window_text.set_draw_color(Color::RGB(0, 0, 0));
    let creator = window_text.texture_creator();

    gl::load_with(|s| video.gl_get_proc_address(s) as *const _);

    if !gl::GetError::is_loaded() {
        return Err(failure::err_msg("glGetError not loaded"));
    }

    unsafe { gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS) };
    check_gl()?;

    gl_register_debug()?;

    kernel::init_gl_funcs(&video);

    let mut interactive_kernel =
        SyncInteractiveKernel::<f32>::create(screen_width, screen_height, is_gl)?;

    let mut fps = FpsCounter::new(1.0);

    let mut framebuffer = 0;
    loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Window {
                    win_event: WindowEvent::Resized(width, height),
                    window_id,
                    ..
                } if window_id == window.id() && width > 0 && height > 0 => {
                    screen_width = width as u32;
                    screen_height = height as u32;
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
                Event::Quit { .. }
                | Event::Window {
                    win_event: WindowEvent::Close,
                    ..
                } => return Ok(()),
                _ => (),
            }
        }

        interactive_kernel.launch()?;
        let img = interactive_kernel.download()?;

        unsafe {
            buffer_blit(
                img.data_gl.expect("gl_display needs OGL texture"),
                &mut framebuffer,
                img.width as i32,
                img.height as i32,
                screen_width as i32,
                screen_height as i32,
            )
        }?;

        window_text.clear();

        let display = format!("{} fps\n{}", fps.value(), interactive_kernel.status());
        render_text(&font, &creator, &mut window_text, &display)?;

        window.gl_swap_window();
        window_text.present();

        fps.tick();
    }
}
