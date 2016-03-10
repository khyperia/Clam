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
        Driver driver;
        driver.MainLoop();
#if CATCH_EXCEPTIONS
    }
    catch (const std::exception &ex)
    {
        std::cout << "Fatal exception:\n" << ex.what() << "\n";
    }
    catch (...)
    {
        std::cout << "Unknown fatal exception\n";
    }
#endif
    return 0;
}
