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
        driver.MainLoop();
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
