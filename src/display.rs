use failure::Error;
pub use glutin::event::VirtualKeyCode as Key;
use glutin::{
    self,
    dpi::PhysicalSize,
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    ContextBuilder,
};
use khygl::{check_gl, gl_register_debug};
use std::ffi::CStr;

pub trait Display: Sized {
    fn setup(size: (usize, usize), dpi: f64) -> Result<Self, Error>;
    fn render(&mut self) -> Result<(), Error>;
    fn resize(&mut self, size: (usize, usize)) -> Result<(), Error>;
    fn key_up(&mut self, key: Key) -> Result<(), Error>;
    fn key_down(&mut self, key: Key) -> Result<(), Error>;
    fn received_character(&mut self, ch: char) -> Result<(), Error>;
}

fn print_name() -> Result<(), Error> {
    unsafe {
        let renderer = CStr::from_ptr(gl::GetString(gl::RENDERER) as _);
        check_gl()?;
        println!("Using engine: {:?}", renderer);
        let shader_version = CStr::from_ptr(gl::GetString(gl::SHADING_LANGUAGE_VERSION) as _);
        check_gl()?;
        println!("Shader version: {:?}", shader_version);
    }
    Ok(())
}

pub fn run_headless<T>(func: impl Fn() -> T) -> Result<T, Error> {
    unsafe {
        let el = EventLoop::new();
        let ctx = ContextBuilder::new().build_headless(&el, PhysicalSize::new(10.0, 10.0))?;
        let ctx_cur = ctx.make_current().map_err(|(_, e)| e)?;
        gl::load_with(|symbol| ctx_cur.get_proc_address(symbol) as *const _);
        print_name()?;
        Ok(func())
    }
}

pub fn run<Disp: Display + 'static>(request_size: (f64, f64)) -> Result<(), Error> {
    let el = EventLoop::new();
    //let vm = el.primary_monitor().video_modes().min().expect("No video modes found");
    let wb = WindowBuilder::new()
        .with_title("clam5")
        //.with_fullscreen(Some(Fullscreen::Exclusive(vm)));
        .with_inner_size(glutin::dpi::LogicalSize::new(
            request_size.0,
            request_size.1,
        ));
    let windowed_context = ContextBuilder::new()
        .with_vsync(true)
        .build_windowed(wb, &el)?;

    let windowed_context = unsafe { windowed_context.make_current().map_err(|(_, e)| e)? };

    let initial_size = windowed_context
        .window()
        .inner_size()
        .to_physical(windowed_context.window().hidpi_factor());

    gl::load_with(|symbol| windowed_context.get_proc_address(symbol) as *const _);

    if !gl::GetError::is_loaded() {
        return Err(failure::err_msg("glGetError not loaded"));
    }

    if cfg!(debug_assertions) {
        unsafe { gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS) };
        check_gl()?;
        gl_register_debug()?;
    }

    print_name()?;

    let dpi = windowed_context.window().hidpi_factor();

    let mut display = Some(Disp::setup(
        (initial_size.width as usize, initial_size.height as usize),
        dpi,
    )?);

    el.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::Resized(logical_size)
                if logical_size.width > 0.0 && logical_size.height > 0.0 =>
            {
                if let Some(ref mut display) = display {
                    let dpi_factor = windowed_context.window().hidpi_factor();
                    let physical = logical_size.to_physical(dpi_factor);
                    handle(display.resize((physical.width as usize, physical.height as usize)));
                    unsafe { gl::Viewport(0, 0, physical.width as i32, physical.height as i32) };
                    windowed_context.resize(logical_size.to_physical(dpi_factor));
                }
            }
            WindowEvent::KeyboardInput { input, .. } => {
                if let Some(ref mut display) = display {
                    if let Some(code) = input.virtual_keycode {
                        match input.state {
                            ElementState::Pressed => handle(display.key_down(code)),
                            ElementState::Released => handle(display.key_up(code)),
                        }
                    }
                }
            }
            WindowEvent::ReceivedCharacter(ch) => {
                if let Some(ref mut display) = display {
                    handle(display.received_character(ch))
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(ref mut display) = display {
                    handle(display.render());
                    handle(windowed_context.swap_buffers().map_err(|e| e.into()));
                }
            }
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            _ => (),
        },
        Event::EventsCleared => {
            if *control_flow == ControlFlow::Exit {
                display = None;
            } else {
                windowed_context.window().request_redraw()
            }
        }
        _ => (),
    })
}

fn handle<T>(res: Result<T, Error>) -> T {
    match res {
        Ok(ok) => ok,
        Err(err) => panic!("{:?}", err),
    }
}
