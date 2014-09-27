#include "client.h"
#include "server.h"
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

int unsafemain(int argc, char** argv)
{
    glutInit(&argc, argv);

    const char* programType = getenv("PROGRAMTYPE");
    if (!programType)
    {
        puts("Envrionment variable PROGRAMTYPE not set, exiting");
        return -1;
    }
    else if (strcmp(programType, "slave") == 0)
    {
        const char* slavePort = getenv("SLAVE_PORT");
        if (!slavePort)
        {
            puts("Envrionment variable SLAVE_PORT not set, exiting");
            return -1;
        }
        const char* slaveWindow = getenv("SLAVE_WINDOW");
        if (!slaveWindow)
        {
            puts("Envrionment variable SLAVE_WINDOW not set, exiting");
            return -1;
        }
        int width, height, xpos, ypos;
        if (sscanf(slaveWindow, "%dx%d+%d+%d", &width, &height, &xpos, &ypos) != 4)
        {
            puts("$SLAVE_WINDOW was not in format %dx%d+%d+%d, exiting");
            return -1;
        }
        const char* slaveFullscreen = getenv("SLAVE_FULLSCREEN");
        bool slaveFullscreenVal = slaveFullscreen && strcmp(slaveFullscreen, "true") == 0;
        client(slavePort, xpos, ypos, width, height, slaveFullscreenVal);
        return 0;
    }
    else if (strcmp(programType, "master") == 0)
    {
        const char* masterKernel = getenv("MASTER_KERNEL");
        if (!masterKernel)
        {
            puts("Envrionment variable MASTER_KERNEL not set, exiting");
            return -1;
        }
        const char* clients = getenv("MASTER_CONNECTIONS");
        if (!clients)
        {
            puts("Environment variable MASTER_CONNECTIONS not set, exiting");
            return -1;
        }
        auto clientVec = split(clients, ';');
        server(masterKernel, clientVec);
        return 0;
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
        unsafemain(argc, argv);
    }
    catch (std::exception const& ex)
    {
        puts("Unhandled exception:");
        puts(ex.what());
    }
}
