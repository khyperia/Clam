#include "server.h"
#include "interop.h"
#include <memory>
#include <set>
#include <fstream>
#include <chrono>
#include <GL/glut.h>
#include "scriptengine.h"

std::shared_ptr<std::vector<std::shared_ptr<CSocket>>> socks = nullptr;
std::shared_ptr<ScriptEngine> scriptengine = nullptr;
float offsetX = 0.f, offsetY = 0.f, zoom = 1.f;
std::set<char> pressedKeys;
std::set<int> pressedSpecialKeys;
auto lastUpdate = std::chrono::steady_clock::now();

bool iskeydown(char key)
{
    return pressedKeys.find(key) != pressedKeys.end();
}

void unsetkey(char key)
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

void keyboardDownFunc(unsigned char ukey, int, int)
{
    char key = static_cast<char>(ukey);
    if (pressedKeys.find(key) != pressedKeys.end())
        return;
    pressedKeys.insert(key);
}

void keyboardUpFunc(unsigned char ukey, int, int)
{
    char key = static_cast<char>(ukey);
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

void server(std::string luaFile, std::vector<std::string> clients)
{
    socks = std::make_shared<std::vector<std::shared_ptr<CSocket>>>();
    atexit(closeSocks);
    for (auto const& arg : clients)
    {
        socks->push_back(std::make_shared<CSocket>(arg));
    }

    std::ifstream inputFile(luaFile);
    if (inputFile.is_open() == false)
        throw std::runtime_error("Lua file \"" + luaFile + "\" didn't exist");
    std::string sourcecode((std::istreambuf_iterator<char>(inputFile)),
            std::istreambuf_iterator<char>());

    scriptengine = std::make_shared<ScriptEngine>(sourcecode);

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
