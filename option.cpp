#include "option.h"
#include "util.h"
#include <stdio.h>

static std::string masterIp;
static std::string hostBindPort;
static std::string windowPos;
static std::string kernelName;
static std::string headless;
static std::string renderOffset;
static std::string vrpn;
static std::string font;
static std::string cudaDeviceNum;

static void TryParseEnv(std::string &ref, const char *varname)
{
    const char *env = getenv(varname);
    if (env)
    {
        ref = env;
    }
}

static void ParseEnvVar()
{
    TryParseEnv(masterIp, "CLAM3_CONNECT");
    TryParseEnv(hostBindPort, "CLAM3_HOST");
    TryParseEnv(windowPos, "CLAM3_WINDOWPOS");
    TryParseEnv(kernelName, "CLAM3_KERNEL");
    TryParseEnv(headless, "CLAM3_HEADLESS");
    TryParseEnv(renderOffset, "CLAM3_RENDEROFFSET");
    TryParseEnv(vrpn, "CLAM3_VRPN");
    TryParseEnv(font, "CLAM3_FONT");
    TryParseEnv(cudaDeviceNum, "CLAM3_DEVICE");
}

static void ParseArg(const std::string &option, const std::string &value)
{
    if (option == "--connect" || option == "-c")
    {
        masterIp = value;
    }
    else if (option == "--host" || option == "-h")
    {
        hostBindPort = value;
    }
    else if (option == "--windowpos" || option == "-w")
    {
        windowPos = value;
    }
    else if (option == "--kernel" || option == "-k")
    {
        kernelName = value;
    }
    else if (option == "--headless")
    {
        headless = value;
    }
    else if (option == "--renderoffset" || option == "-r")
    {
        renderOffset = value;
    }
    else if (option == "--vrpn")
    {
        vrpn = value;
    }
    else if (option == "--font")
    {
        font = value;
    }
    else if (option == "--device")
    {
        cudaDeviceNum = value;
    }
    else
    {
        std::cout << "Unknown option " << option << "\n";
    }
}

void ParseCmdline(int argc, char **argv)
{
    ParseEnvVar();
    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        if (i >= argc)
        {
            std::cout << "Option " << option << " needs a value\n";
            continue;
        }
        std::string value(argv[i]);
        ParseArg(option, value);
    }
}

std::string MasterIp()
{
    return masterIp;
}

std::string HostBindPort()
{
    return hostBindPort;
}

bool IsCompute()
{
    return !MasterIp().empty() || HostBindPort().empty();
}

bool IsUserInput()
{
    return !HostBindPort().empty() || MasterIp().empty();
}

std::string WindowPos()
{
    return windowPos;
}

std::string KernelName()
{
    return kernelName;
}

int Headless(int *numTimes)
{
    if (headless.empty())
    {
        return -1;
    }
    int headlessCount = 0;
    int numScanned = sscanf(headless.c_str(), "%d,%d", &headlessCount, numTimes);
    if (numScanned < 2)
    {
        *numTimes = 0;
    }
    if (numScanned == 0)
        return -1;
    return headlessCount;
}

bool RenderOffset(int *shiftx, int *shifty)
{
    if (renderOffset.empty())
    {
        return false;
    }
    if (sscanf(renderOffset.c_str(), "%dx%d", shiftx, shifty) != 2)
    {
        std::cout << "Couldn't parse render offset, assuming no offset\n";
        return false;
    }
    return true;
}

std::string VrpnName()
{
    return vrpn;
}

std::string FontName()
{
    return font;
}

int CudaDeviceNum()
{
    if (cudaDeviceNum.empty())
    {
        return 0;
    }
    int num;
    if (sscanf(cudaDeviceNum.c_str(), "%d", &num) != 1)
    {
        std::cout << "Couldn't parse cuda device number, assuming 0\n";
        num = 0;
    }
    return num;
}
