use crate::check_gl;
use crate::fps_counter::FpsCounter;
use crate::gl_register_debug;
use crate::interactive::SyncInteractiveKernel;
use crate::kernel;
use crate::render_text::TextRenderer;
use crate::render_texture::TextureRenderer;
use failure::err_msg;
use failure::Error;
use gl;
use sdl2::event::Event;
use sdl2::event::WindowEvent;
use sdl2::init;
use sdl2::pixels::Color;

pub fn gl_display(mut screen_width: u32, mut screen_height: u32) -> Result<(), Error> {
    let is_gl = true;
    let sdl = init().map_err(err_msg)?;
    let video = sdl.video().map_err(err_msg)?;
    let mut event_pump = sdl.event_pump().map_err(err_msg)?;

    video.gl_attr().set_context_flags().debug().set();

    let window = video
        .window("clam5", screen_width, screen_height)
        //.resizable()
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

    let mut interactive_kernel =
        SyncInteractiveKernel::<f32>::create(screen_width, screen_height, is_gl)?;

    let texture_renderer = TextureRenderer::new();
    let text_renderer = TextRenderer::new(Color::RGB(255, 192, 192));

    let mut fps = FpsCounter::new(1.0);

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
                    println!("Resizing kernel");
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

        texture_renderer.render(
            img.data_gl.expect("gl_display needs OGL texture"),
            0.0,
            0.0,
            1.0,
            1.0,
        )?;

        let display = format!("{} fps\n{}", fps.value(), interactive_kernel.status());
        text_renderer.render(&texture_renderer, &display, screen_width, screen_height)?;

        window.gl_swap_window();

        check_gl()?;

        fps.tick();
    }
}
