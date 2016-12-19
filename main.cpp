#include "driver.h"
#include "option.h"

#ifdef main
#undef main
#endif

#define CATCH_EXCEPTIONS 0

// Legacy code for reference on UI design
/*
TryParseEnv(masterIp, "CLAM3_CONNECT");
TryParseEnv(hostBindPort, "CLAM3_HOST");
TryParseEnv(windowPos, "CLAM3_WINDOWPOS");
TryParseEnv(kernelName, "CLAM3_KERNEL");
TryParseEnv(headless, "CLAM3_HEADLESS");
TryParseEnv(renderOffset, "CLAM3_RENDEROFFSET");
TryParseEnv(vrpn, "CLAM3_VRPN");
TryParseEnv(font, "CLAM3_FONT");
TryParseEnv(cudaDeviceNum, "CLAM3_DEVICE");
TryParseEnv(saveProgress, "CLAM3_SAVEPROGRESS");
if (option == "--connect" || option == "-c")
else if (option == "--host" || option == "-h")
else if (option == "--windowpos" || option == "-w")
else if (option == "--kernel" || option == "-k")
else if (option == "--headless")
else if (option == "--renderoffset" || option == "-r")
else if (option == "--vrpn")
else if (option == "--font")
else if (option == "--device")
else if (option == "--saveprogress")

else if (keycode == SDLK_g)
editValue.DofAmount = editValue.DofAmount > 0 ? 0 : 0.005f;

if (keycode == SDLK_BACKSLASH)
std::cout << "Enabling VRPN" << std::endl;
VrpnHelp *vrpn;

vrpn->MainLoop();
pos.x = vrpn->pos.x * 1;
pos.y = (vrpn->pos.z - 1.5) * -1;
zoom = exp(vrpn->pos.y);
*/

int parse_size(const std::string &str, int *width, int *height)
{
    if (str == "hd")
    {
        *width = 1920;
        *height = 1080;
        return 0;
    }
    if (str == "mon")
    {
        *width = 2560;
        *height = 1440;
        return 0;
    }
    if (str == "4k")
    {
        *width = 3840;
        *height = 2160;
        return 0;
    }
    if (str == "help")
    {
        std::cout << "Size options:" << std::endl;
        std::cout << "'hd' = 1920x1080" << std::endl;
        std::cout << "'mon' = 2560x1440" << std::endl;
        std::cout << "'4k' = 3840x2160" << std::endl;
        std::cout << "[int]x[int] = width x height" << std::endl;
        std::cout << "[int] = (val*16/9) x val" << std::endl;
        return 1;
    }
    if (sscanf(str.c_str(), "%dx%d", width, height) == 2)
    {
        return 0;
    }
    if (sscanf(str.c_str(), "%d", height) != 2)
    {
        *width = *height * 16 / 9;
        return 0;
    }
    std::cout << "size option has incorrect format" << std::endl;
    return 1;
}

int main(int argc, char **argv)
{
#if CATCH_EXCEPTIONS
    try
    {
#endif
    UserOptions options;
    options.Add("kernel", "mandelbrot");
    options.Add("gpu", "0");
    options.Add("save", "");
    options.Add("frames", "100");
    options.Add("movie", "0");
    options.Add("movie_loop", "false");
    options.Add("size", "600x400");
    std::string cmdline_error = options.Parse(argc, argv);
    if (!cmdline_error.empty())
    {
        std::cout << cmdline_error << std::endl;
        return 1;
    }

    int width = 0;
    int height = 0;
    if (parse_size(options.Get("size"), &width, &height))
    {
        return 1;
    }

    CudaContext::Init();
    CudaContext ctx(fromstring<int>(options.Get("gpu")));
    ctx.SetCurrent();

    auto render_filename = options.Get("save");
    auto movie = fromstring<int>(options.Get("movie"));
    if (render_filename.empty())
    {
        RealtimeRender driver(std::move(ctx),
                              KernelConfiguration(options.Get("kernel")),
                              width,
                              height,
                              "/usr/share/fonts/TTF/DejaVuSansMono.ttf");
        driver.Run();
    }
    else if (movie > 0)
    {
        auto num_frames = fromstring<int>(options.Get("frames"));
        auto loop = options.Get("movie_loop") != "false";
        MovieRender driver(std::move(ctx),
                           KernelConfiguration(options.Get("kernel")),
                           (size_t)width,
                           (size_t)height,
                           num_frames,
                           movie,
                           loop,
                           "movie.clam3",
                           render_filename);
        driver.Run();
    }
    else
    {
        auto num_frames = fromstring<int>(options.Get("frames"));
        HeadlessRender driver(std::move(ctx),
                              KernelConfiguration(options.Get("kernel")),
                              (size_t)width,
                              (size_t)height,
                              num_frames,
                              "settings.clam3",
                              render_filename);
        driver.Run();
    }
#if CATCH_EXCEPTIONS
    }
    catch (const std::exception &ex)
    {
        std::cout << "Fatal exception:" << std::endl << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown fatal exception" << std::endl;
    }
#endif
    return 0;
}
