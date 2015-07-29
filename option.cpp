#include "option.h"
#include "util.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>

static std::string platform;
static std::string device;
static std::string masterIp;
static std::string hostBindPort;
static std::string windowPos;
static std::string kernelName;
static std::string headless;
static std::string renderOffset;
static std::string dumpBinary;
static bool throwExceptions;

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
    TryParseEnv(platform, "CLAM3_PLATFORM");
    TryParseEnv(device, "CLAM3_DEVICE");
    TryParseEnv(masterIp, "CLAM3_CONNECT");
    TryParseEnv(hostBindPort, "CLAM3_HOST");
    TryParseEnv(windowPos, "CLAM3_WINDOWPOS");
    TryParseEnv(kernelName, "CLAM3_KERNEL");
    TryParseEnv(headless, "CLAM3_HEADLESS");
    TryParseEnv(renderOffset, "CLAM3_RENDEROFFSET");
    TryParseEnv(dumpBinary, "CLAM3_DUMPBIN");
}

static void ParseArg(const std::string &option, const std::string &value)
{
    if (option == "--platform" || option == "-p")
    {
        platform = value;
    }
    else if (option == "--device" || option == "-d")
    {
        device = value;
    }
    else if (option == "--connect" || option == "-c")
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
    else if (option == "--dumpbin")
    {
        dumpBinary = value;
    }
    else if (option == "--debug" && value == "throw")
    {
        throwExceptions = true;
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
            std::cout << "Option " << option << " needs a value";
        }
        std::string value(argv[i]);
        ParseArg(option, value);
    }
}

std::string PlatformName()
{
    return platform;
}

std::string DeviceName()
{
    return device;
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

int Headless()
{
    if (headless.empty())
    {
        return -1;
    }
    try
    {
        return fromstring<int>(headless);
    }
    catch (const std::exception &ex)
    {
        std::cout << "Couldn't parse headless option, assuming no headless" << std::endl;
        return -1;
    }
}

bool RenderOffset(int *shiftx, int *shifty)
{
    if (renderOffset.empty())
    {
        return false;
    }
    if (sscanf(renderOffset.c_str(), "%dx%d", shiftx, shifty) != 2)
    {
        std::cout << "Couldn't parse render offset, assuming no offset" << std::endl;
        return false;
    }
    return true;
}

std::string DumpBinary()
{
    return dumpBinary;
}

bool ThrowExceptions()
{
    return throwExceptions;
}
