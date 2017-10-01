extern crate glium;
extern crate ocl;
extern crate ocl_core;
extern crate png;

mod display;
mod input;
mod mandelbox_cfg;
mod kernel;
mod settings;

use std::sync::mpsc;
use std::thread;
use std::env::args;

fn render() {
    match kernel::headless(1000, 1000, 2) {
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
        send_image,
        recv_screen_event,
    ) {
        Ok(()) => (),
        Err(err) => println!("{}", err),
    });

    match display::display(width, height, recv_image, send_screen_event) {
        Ok(()) => (),
        Err(err) => println!("{}", err),
    };
}

fn main() {
    for arg in args() {
        if arg == "--render" {
            render();
            return;
        }
    }
    interactive();
}
