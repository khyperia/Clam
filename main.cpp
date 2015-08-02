#include "option.h"
#include "driver.h"
#include <iostream>

#undef main
int main(int argc, char **argv)
{
    try
    {
        ParseCmdline(argc, argv);
        Driver driver;
        bool isCompute = IsCompute();
        Uint32 ticks = SDL_GetTicks();
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
    catch (const cl::Error &ex)
    {
        std::cout << "Fatal OpenCL exception (" << ex.err() << "):" << std::endl;
        std::cout << ex.what() << std::endl;
        return 1;
    }
    catch (const std::exception &ex)
    {
        std::cout << "Fatal exception:" << std::endl;
        std::cout << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
