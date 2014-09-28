#include "server.h"
#include "interop.h"
#include <memory>
#include <set>
#include <fstream>
#include <chrono>
#include <GL/glut.h>
#include "scriptengine.h"

std::shared_ptr<std::vector<std::shared_ptr<CSocket>>> socks;
std::shared_ptr<ScriptEngine> scriptengine;
float offsetX = 0.f, offsetY = 0.f, zoom = 1.f;
std::set<unsigned char> pressedKeys;
std::set<int> pressedSpecialKeys;
auto lastUpdate = std::chrono::steady_clock::now();

bool iskeydown(unsigned char key)
{
    return pressedKeys.find(key) != pressedKeys.end();
}

void unsetkey(unsigned char key)
{
    auto loc = pressedKeys.find(key);
    if (loc != pressedKeys.end())
        pressedKeys.erase(loc);
}

std::shared_ptr<std::vector<std::shared_ptr<CSocket>>> getSocks()
{
    return socks;
}

void idleFuncServer()
{
    auto thisUpdate = std::chrono::steady_clock::now();
    auto frameSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(thisUpdate - lastUpdate).count() / 1000.f;
    lastUpdate = thisUpdate;

    scriptengine->Update(frameSeconds);

    glutPostRedisplay();
}

void keyboardDownFunc(unsigned char key, int, int)
{
    if (pressedKeys.find(key) != pressedKeys.end())
        return;
    pressedKeys.insert(key);
}

void keyboardUpFunc(unsigned char key, int, int)
{
    if (pressedKeys.find(key) == pressedKeys.end())
        return;
    pressedKeys.erase(pressedKeys.find(key));
}

void specialDownFunc(int key, int, int)
{
    if (pressedSpecialKeys.find(key) != pressedSpecialKeys.end())
        return;
    pressedSpecialKeys.insert(key);
}

void specialUpFunc(int key, int, int)
{
    if (pressedSpecialKeys.find(key) == pressedSpecialKeys.end())
        return;
    pressedSpecialKeys.erase(pressedSpecialKeys.find(key));
}

void displayFuncServer()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
}

void closeSocks()
{
    for (auto& sock : *socks)
    {
        sock->Send<unsigned int>({MessageKill});
        sock = nullptr;
    }
    socks = nullptr;
    puts("Closed socks");
}

void server(std::string kernelFile, std::vector<std::string> clients)
{
    socks = std::make_shared<std::vector<std::shared_ptr<CSocket>>>();
    std::ifstream inputFile(kernelFile);
    if (inputFile.is_open() == false)
        throw std::runtime_error("Kernel file \"" + kernelFile + "\" didn't exist");
    std::string sourcecode((std::istreambuf_iterator<char>(inputFile)),
            std::istreambuf_iterator<char>());

    auto start = sourcecode.find("CLAMSCRIPTSTART");
    auto end = sourcecode.find("CLAMSCRIPTEND");
    if (start == std::string::npos || end == std::string::npos)
        throw std::runtime_error("Kernel file did not have CLAMSCRIPTSTART/CLAMSCRIPTEND tags");
    start += strlen("CLAMSCRIPTSTART");
    std::string scriptcode = sourcecode.substr(start, end - start);
    scriptengine = std::make_shared<ScriptEngine>(scriptcode);
    
    for (auto arg : clients)
    {
        auto sock = std::make_shared<CSocket>(arg);
        sock->Send<unsigned int>({MessageKernelSource});
        sock->SendStr(sourcecode);
        sock->Recv<uint8_t>(1);
        socks->push_back(sock);
    }

    atexit(closeSocks);

    glutInitDisplayMode(GLUT_DOUBLE);
    glutCreateWindow("Clam2 Server");

    glutKeyboardUpFunc(keyboardUpFunc);
    glutKeyboardFunc(keyboardDownFunc);
    glutSpecialUpFunc(specialUpFunc);
    glutSpecialFunc(specialDownFunc);
    glutIdleFunc(idleFuncServer);
    glutDisplayFunc(displayFuncServer);
    glutMainLoop();
}
