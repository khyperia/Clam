Clam3
===

This thing has been rewritten so many times I don't even want to think about it.

Anyway, this project is a GPU fractal renderer. It [makes stuff](http://imgur.com/a/rmM4v).

---

Supported platforms: Linux primarily (the main dev's box is Arch), it has built on Windows before, and *probably* will on Mac.

Requirements: CUDA (all kernels are written in CUDA C), SDL2, SDL2_ttf, optionally VRPN for an IR tracking system at the author's university.

---

What major types are and where they are located:

Entry point (main.cpp): collects configuration and invokes the appropriate driver.

Driver, e.g. RealtimeRender (driver.h/cpp): Constructs and coordinates interactions between various modules. Kind of a hairy mess due to things like async rendering being centered here, but is the heart of the app.

CudaContext (cudaContext.h/cpp): Responsible for creating/managing the CUDA environment. Any Cuda calls should be wrapped into a call to .Run() on a CudaContext.

KernelConfig (kernelConfig.h/cpp): A mapping from name to kernel parameters (like variables and controls)

GpuKernel (kernel.h/cpp): A single GPU kernel, associated stream/memory and any global variables it has.

SettingCollection (kernelSetting.h/cpp): A big bag of name/value pairs. Discussed later.

KernelControl (kernelControl.h/cpp): Takes a SettingCollection and applies it to a kernel.

UiSetting (uiSetting.h/cpp): Takes SDL keypresses and applies them to a SettingCollection.

RenderTarget (display.h/cpp): Objects that pixels can be rendered to. SDL window, file, etc.

CUDA kernels (*.cu): Guts of the rendering, all math is performed here.

---

Overview of behavior:

The entry point is run, and all "constant" configuration is performed. This involves:

* Opening an SDL window, loading keyframes from a file, etc.
* Figuring out what kernel to run
* Connecting to or launching remote instances
* Creating and hooking up all components

Then, an "event loop" is started (may be an actual UI event loop, or a simple while loop, depending on configuration)

All "runtime" parameters are stored inside a SettingCollection bag. "Runtime" in this case may be generally viewed as parameters that *may* change between individual kernel invocations. This involves fractal parameters, camera location, etc.

The following happens in a loop:

* UiSetting objects are queried for inputs, which mutate the SettingCollection bag. These inputs may be network connections, keyboard input, etc.
* KernelControl objects are then called to copy various bits of the SettingCollection bag into specific GpuKernelVar structures.
* The actual GpuKernel is then invoked, starting a GPU computation with the settings in the GpuKernelVar structures.
* When completed, an (async) callback happens with the rendered data, which is then handed to the RenderTarget to display.

Of course, each step may be subtly different for various configurations, for example, rendering to a file instead of a window, or multiple kernels being invoked simultaneously or asynchronously.
