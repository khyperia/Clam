#include "option.h"
#include "driver.h"

#ifdef main
#undef main
#endif

int main(int argc, char **argv)
{
    ParseCmdline(argc, argv);
    Driver driver;
    bool isCompute = IsCompute();
    Uint32 ticks = SDL_GetTicks();
    try
    {
        while (driver.RunFrame())
        {
            if (!isCompute)
            {
                Uint32 newTicks = SDL_GetTicks();
                Uint32 wait = newTicks - ticks;
                ticks = newTicks;
                const Uint32 delay = 1000 / 60;
                if (wait < delay)
                {
                    SDL_Delay(delay - wait);
                }
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cout << "Fatal exception:\n" << ex.what() << "\n";
    }
    return 0;
}
