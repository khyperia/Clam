use display::{Image, ScreenEvent};
use failure::Error;
use input::Input;
use kernel::Kernel;
use kernel_compilation;
use settings::Settings;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

pub struct InteractiveKernel {
    image_stream: mpsc::Receiver<Image>,
    screen_events: mpsc::Sender<ScreenEvent>,
}

pub enum DownloadResult {
    NoMoreImages,
    NoneAtPresent,
    Image(Image),
}

impl InteractiveKernel {
    pub fn create(
        width: u32,
        height: u32,
        settings_input: Arc<Mutex<(Settings, Input)>>,
    ) -> Result<Self, Error> {
        let (screen_send, screen_recv) = mpsc::channel();
        let (image_send, image_recv) = mpsc::sync_channel(2);

        kernel_compilation::watch_src(screen_send.clone())?;

        thread::spawn(move || {
            let kernel = Kernel::create(width, height, &mut settings_input.lock().unwrap().0).unwrap();
            match Self::run_thread(kernel, &screen_recv, &image_send, &settings_input) {
                Ok(()) => (),
                Err(err) => panic!("Error in kernel thread: {}", err),
            }
        });

        Ok(Self {
            image_stream: image_recv,
            screen_events: screen_send,
        })
    }

    pub fn download(&self) -> DownloadResult {
        let mut result = DownloadResult::NoneAtPresent;
        loop {
            let image = match self.image_stream.try_recv() {
                Ok(image) => image,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return DownloadResult::NoMoreImages,
            };
            result = DownloadResult::Image(image);
        }
        result
    }

    pub fn resize(&self, width: u32, height: u32) -> bool {
        match self.screen_events.send(ScreenEvent::Resize(width, height)) {
            Ok(()) => true,
            Err(mpsc::SendError(_)) => false,
        }
    }

    fn run_thread(
        mut kernel: Kernel,
        screen_events: &mpsc::Receiver<ScreenEvent>,
        send_image: &mpsc::SyncSender<Image>,
        settings_input: &Arc<Mutex<(Settings, Input)>>,
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
                    ScreenEvent::KernelChanged => {
                        let mut locked = settings_input.lock().unwrap();
                        let (ref mut settings, _) = *locked;
                        kernel.rebuild(settings)?
                    }
                }
            }

            let settings = {
                let mut locked = settings_input.lock().unwrap();
                let (ref mut settings, ref mut input) = *locked;
                input.integrate(settings);
                if settings.check_rebuild() {
                    kernel.rebuild(settings)?;
                }
                (*settings).clone()
            };
            kernel.run(&settings)?;
            let image = kernel.download()?;
            match send_image.send(image) {
                Ok(()) => (),
                Err(mpsc::SendError(_)) => return Ok(()),
            };
        }
    }
}
