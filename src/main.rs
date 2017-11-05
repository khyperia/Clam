extern crate glium;
extern crate ocl;
extern crate ocl_core;
extern crate png;

mod display;
mod input;
mod kernel;
mod mandelbox_cfg;
mod progress;
mod settings;

use std::sync::mpsc;
use std::thread;
use std::env::args;

fn try_render(args: &[String]) -> Result<(), Box<std::error::Error>> {
    if args.len() == 2 {
        let rpp = args[1].parse()?;
        match &*args[0] {
            "8k" => kernel::headless(3840 * 2, 2160 * 2, rpp),
            "4k" => kernel::headless(3840, 2160, rpp),
            "1080p" | "2k" => kernel::headless(1920, 1080, rpp),
            pix => kernel::headless(pix.parse()?, pix.parse()?, rpp),
        }
    } else {
        kernel::headless(args[0].parse()?, args[1].parse()?, args[2].parse()?)
    }
}

fn render(args: &[String]) {
    match try_render(args) {
        Ok(()) => (),
        Err(err) => println!("{}", err),
    }
}

fn interactive() {
    let (send_image, recv_image) = mpsc::channel();
    let (send_screen_event, recv_screen_event) = mpsc::channel();
    let width = 200;
    let height = 200;

    thread::spawn(move || match kernel::interactive(
        width,
        height,
        &send_image,
        &recv_screen_event,
    ) {
        Ok(()) => (),
        Err(err) => println!("{}", err),
    });

    match display::display(width, height, &recv_image, &send_screen_event) {
        Ok(()) => (),
        Err(err) => println!("{}", err),
    };
}

fn main() {
    let arguments = args().skip(1).collect::<Vec<_>>();
    if arguments.len() > 2 && arguments[0] == "--render" {
        render(&arguments[1..]);
    } else {
        interactive();
    }
}
