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

MkNv(float, Scale, 0.25f);
MkNv(float, FoldingLimit, 0.25f);
MkNv(float, FixedRadius2, 0.25f);
MkNv(float, MinRadius2, 0.25f);
MkNv(float, DeRotationAmount, 0.25f);
MkNv(float, DeRotationAxisX, 0.25f);
MkNv(float, DeRotationAxisY, 0.25f);
MkNv(float, DeRotationAxisZ, 0.25f);
MkNv(float, DofAmount, 0.01f);
MkNv(float, FovAbberation, -0.1f);
MkNv(float, LightPosX, 0.5f);
MkNv(float, LightPosY, 0.5f);
MkNv(float, LightPosZ, 0.5f);
MkNv(float, LightSize, -0.2f);
MkNv(int, WhiteClamp, 0);
MkNv(float, LightBrightnessHue, 0.125f);
MkNv(float, LightBrightnessSat, -0.25f);
MkNv(float, LightBrightnessVal, -0.25f);
MkNv(float, AmbientBrightnessHue, 0.125f);
MkNv(float, AmbientBrightnessSat, -0.25f);
MkNv(float, AmbientBrightnessVal, -0.25f);
MkNv(float, ReflectHue, 0.125f);
MkNv(float, ReflectSat, -0.25f);
MkNv(float, ReflectVal, -0.25f);
MkNv(int, MaxIters, 0);
MkNv(float, Bailout, -0.5f);
MkNv(float, DeMultiplier, 0.125f);
MkNv(int, RandSeedInitSteps, 0);
MkNv(float, MaxRayDist, -1.0f);
MkNv(int, MaxRaySteps, 0);
MkNv(int, NumRayBounces, 0);
MkNv(float, QualityFirstRay, -0.5f);
MkNv(float, QualityRestRay, -0.5f);
MkNv(int, ItersPerKernel, 0);

else if (keycode == SDLK_g)
editValue.DofAmount = editValue.DofAmount > 0 ? 0 : 0.005f;

if (keycode == SDLK_BACKSLASH)
std::cout << "Enabling VRPN" << std::endl;
VrpnHelp *vrpn;

vrpn->MainLoop();
pos.x = vrpn->pos.x * 1;
pos.y = (vrpn->pos.z - 1.5) * -1;
zoom = exp(vrpn->pos.y);

if (keycode == SDLK_u || keycode == SDLK_o)
    juliaEnabled = !juliaEnabled;

if (juliaEnabled && keycode == SDLK_i)
    juliaPos.y -= time * moveSpeed;
else if (juliaEnabled && keycode == SDLK_k)
    juliaPos.y += time * moveSpeed;
else if (juliaEnabled && keycode == SDLK_j)
    juliaPos.x -= time * moveSpeed;
else if (juliaEnabled && keycode == SDLK_l)
    juliaPos.x += time * moveSpeed;
else if (juliaEnabled && keycode == SDLK_n)
    moveSpeed *= 1 + time;
else if (juliaEnabled && keycode == SDLK_m)
    moveSpeed /= 1 + time;

double weight = 1 / (fpsAverage + 1);
fpsAverage = (timePassed + fpsAverage * weight) / (weight + 1);
timeSinceLastTitle += timePassed;
while (timeSinceLastTitle > 1.0)
{
    SDL_Delay(1);
    SDL_SetWindowTitle(window, ("Clam3 - " + tostring(1 / fpsAverage) + " fps").c_str());
    timeSinceLastTitle--;
}
*/

// TODO: Pull window pos from WindowPos()
// TODO: Renderstate filename [kernel].[shiftx]x[shifty].t[time].gpu[gpu].clam3
// TODO: Headless default: renderstate then config state
// TODO: Renderstate default: renderstate then config state
// TODO: Headless frames/times

int main(int argc, char **argv)
{
#if CATCH_EXCEPTIONS
    try
    {
#endif
    UserOptions options;
    options.Add("kernel", "mandelbrot");
    options.Add("gpu", "0");
    std::string cmdline_error = options.Parse(argc, argv);
    if (!cmdline_error.empty())
    {
        std::cout << cmdline_error << std::endl;
        return 1;
    }

    CudaContext::Init();
    CudaContext ctx(fromstring<int>(options.Get("gpu")));
    RealtimeRender driver(std::move(ctx),
                          KernelConfiguration(options.Get("kernel")),
                          600,
                          400,
                          "/usr/share/fonts/TTF/DejaVuSansMono.ttf");
    driver.StartLoop(1);
    while (true)
    {
        if (!driver.Tick())
            break;
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
