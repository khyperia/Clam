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
static std::string saveProgress;

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
    TryParseEnv(saveProgress, "CLAM3_SAVEPROGRESS");
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
    else if (option == "--saveprogress")
    {
        saveProgress = value;
    }
    else
    {
        std::cout << "Unknown option " << option << std::endl;
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
            std::cout << "Option " << option << " needs a value" << std::endl;
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
    return (!HostBindPort().empty() || MasterIp().empty()) && headless.empty();
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
        *numTimes = 1;
    }
    if (numScanned == 0)
    {
        std::cout << "Couldn't parse headless count, assuming not headless" << std::endl;
        return -1;
    }
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
        std::cout << "Couldn't parse render offset, assuming no offset" << std::endl;
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

std::vector<int> CudaDeviceNums()
{
    std::vector<int> result;
    if (cudaDeviceNum.empty())
    {
        result.push_back(0);
        return result;
    }
    // http://stackoverflow.com/questions/1894886
    std::stringstream ss(cudaDeviceNum);
    int num;
    while (ss >> num)
    {
        result.push_back(num);
        if (ss.peek() == ',')
        {
            ss.ignore();
        }
    }
    if (result.size() == 0)
    {
        throw std::runtime_error("Bad device parameter: no devices specified");
    }
    return result;
}

bool DoSaveProgress()
{
    return saveProgress.empty() || saveProgress == "true" || saveProgress == "1";
}
