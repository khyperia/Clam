use display::{Image, ScreenEvent};
use failure::Error;
use input::Input;
use kernel::{Kernel, KernelImage};
use settings::Settings;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

pub struct InteractiveKernel {
    kernel_data: Arc<KernelImage>,
    screen_events: mpsc::Sender<ScreenEvent>,
}

impl InteractiveKernel {
    pub fn new(
        width: u32,
        height: u32,
        settings_input: Arc<Mutex<(Settings, Input)>>,
    ) -> Result<Self, Error> {
        let (screen_send, screen_recv) = mpsc::channel();
        let (send, recv) = mpsc::sync_channel(0);

        thread::spawn(move || {
            let kernel = Kernel::new(width, height, &settings_input.lock().unwrap().0).unwrap();
            send.send(kernel.alias_data())
                .expect("send kernel data alias");
            match Self::run_thread(kernel, screen_recv, settings_input) {
                Ok(()) => (),
                Err(err) => println!("Error in kernel thread: {}", err),
            }
        });

        let kernel_data = recv.recv().expect("recv kernel data alias");

        Ok(Self {
            kernel_data,
            screen_events: screen_send,
        })
    }

    pub fn download(&self) -> Result<Image, Error> {
        self.kernel_data.download()
    }

    pub fn resize(&self, width: u32, height: u32) -> bool {
        match self.screen_events.send(ScreenEvent::Resize(width, height)) {
            Ok(()) => true,
            Err(mpsc::SendError(_)) => false,
        }
    }

    fn run_thread(
        mut kernel: Kernel,
        screen_events: mpsc::Receiver<ScreenEvent>,
        settings_input: Arc<Mutex<(Settings, Input)>>,
    ) -> Result<(), Error> {
        loop {
            loop {
                let event = match screen_events.try_recv() {
                    Ok(event) => event,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
                };

                match event {
                    ScreenEvent::Resize(width, height) => kernel.resize(width, height)?,
                }
            }

            let settings = {
                let mut locked = settings_input.lock().unwrap();
                let (ref mut settings, ref mut input) = *locked;
                input.integrate(settings);
                if settings.check_rebuild() {
                    kernel.rebuild_self(settings)?;
                }
                (*settings).clone()
            };
            kernel.run(&settings)?;
            kernel.sync_renderer()?;
        }
    }
}
