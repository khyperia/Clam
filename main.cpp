#include "option.h"
#include "driver.h"

#ifdef main
#undef main
#endif

#define CATCH_EXCEPTIONS 0

int main(int argc, char **argv)
{
#if CATCH_EXCEPTIONS
    try
    {
#endif
    ParseCmdline(argc, argv);
    CudaContext::Init();
    CudaContext ctx(0);
    RealtimeRender driver(std::move(ctx),
                          KernelConfiguration("mandelbrot"),
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
