#include "client.h"
#include "server.h"
#include "echo.h"
#include <GL/glut.h>

std::vector<std::string> split(std::string str, char sep)
{
    std::vector<std::string> result;
    for (auto index = str.find(sep); index != std::string::npos; index = str.find(sep))
    {
        result.push_back(str.substr(0, index));
        str = str.substr(index + 1, str.length() - (index + 1));
    }
    result.push_back(str);
    return result;
}

std::string sgetenv(const char* name)
{
    const char* result = getenv(name);
    if (!result)
        throw std::runtime_error("Environment variable " + std::string(name) + " not set");
    return result;
}

int unsafemain(int argc, char** argv)
{
    glutInit(&argc, argv);

    auto programType = sgetenv("PROGRAMTYPE");
    if (programType == "slave")
    {
        auto slavePort = sgetenv("SLAVE_PORT");
        auto slaveWindow = sgetenv("SLAVE_WINDOW");
        int width, height, xpos, ypos;
        if (sscanf(slaveWindow.c_str(), "%dx%d%d%d", &width, &height, &xpos, &ypos) != 4)
        {
            puts("$SLAVE_WINDOW was not in format [width]x[height][-+][posX][-+][posY], exiting");
            return -1;
        }
        auto slaveRendercoords = sgetenv("SLAVE_RENDERCOORDS");
        int renderX, renderY;
        if (sscanf(slaveRendercoords.c_str(), "%d%d", &renderX, &renderY) != 2)
        {
            puts("$SLAVE_RENDERCOORDS was not in format [-+][num][-+][num], exiting");
            return -1;
        }
        const char* slaveFullscreen = getenv("SLAVE_FULLSCREEN");
        bool slaveFullscreenVal = slaveFullscreen && strcmp(slaveFullscreen, "true") == 0;
        const char* keepProcessAliveStr = getenv("SLAVE_KEEPALIVE");
        bool keepProcessAlive = keepProcessAliveStr && strcmp(keepProcessAliveStr, "true") == 0;
        client(slavePort.c_str(),
                xpos, ypos, width, height,
                renderX, renderY,
                slaveFullscreenVal,
                keepProcessAlive);
        return 0;
    }
    else if (programType == "master")
    {
        auto masterLua = sgetenv("MASTER_LUA");
        auto clientVec = split(sgetenv("MASTER_CONNECTIONS"), '~');
        server(masterLua, clientVec);
        return 0;
    }
    else if (programType == "echo")
    {
        auto incomingPort = sgetenv("ECHO_PORT");
        auto outgoingIps = split(sgetenv("ECHO_CONNECTIONS"), '~');
        const char* keepProcessAliveStr = getenv("ECHO_KEEPALIVE");
        bool keepProcessAlive = keepProcessAliveStr && strcmp(keepProcessAliveStr, "true") == 0;
        echo(incomingPort.c_str(), outgoingIps, keepProcessAlive);
        // return 0;
    }
    else
    {
        puts("Unknown $PROGRAMTYPE, exiting");
        return -1;
    }
}

int main(int argc, char** argv)
{
    try
    {
        return unsafemain(argc, argv);
    }
    catch (std::exception const& ex)
    {
        puts(("Unhandled exception (PROGRAMTYPE = " +
                    std::string(getenv("PROGRAMTYPE")) + "):").c_str());
        puts(ex.what());
        return -1;
    }
}
