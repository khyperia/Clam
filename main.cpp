#include "option.h"
#include "driver.h"

#ifdef main
#undef main
#endif

int main(int argc, char **argv)
{
    try
    {
        ParseCmdline(argc, argv);
        Driver driver;
        Uint32 ticks = SDL_GetTicks();
        double time = 1.0;
        while (driver.RunFrame(time))
        {
            SDL_Delay(1);
            Uint32 newTicks = SDL_GetTicks();
            time = (newTicks - ticks) / 1000.0;
            ticks = newTicks;
        }
    }
    catch (const std::exception &ex)
    {
        std::cout << "Fatal exception:\n" << ex.what() << "\n";
    }
    catch (...)
    {
        std::cout << "Unknown fatal exception\n";
    }
    return 0;
}
