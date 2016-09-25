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
        std::cout << "Fatal exception:" << std::endl << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown fatal exception" << std::endl;
    }
#endif
    return 0;
}
