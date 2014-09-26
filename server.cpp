#include "server.h"
#include "interop.h"
#include <memory>
#include <set>
#include <fstream>
#include <chrono>
#include <GL/glut.h>
#include "scriptengine.h"

using boost::asio::ip::tcp;
std::shared_ptr<std::vector<std::shared_ptr<tcp::socket>>> socks;
std::shared_ptr<ScriptEngine> scriptengine;
float offsetX = 0.f, offsetY = 0.f, zoom = 1.f;
std::set<unsigned char> pressedKeys;
std::set<int> pressedSpecialKeys;
auto lastUpdate = std::chrono::steady_clock::now();

bool iskeydown(unsigned char key)
{
    return pressedKeys.find(key) != pressedKeys.end();
}

std::shared_ptr<std::vector<std::shared_ptr<tcp::socket>>> getSocks()
{
    return socks;
}

void idleFuncServer()
{
    auto thisUpdate = std::chrono::steady_clock::now();
    auto frameSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(thisUpdate - lastUpdate).count() / 1000.f;
    lastUpdate = thisUpdate;

    if (scriptengine && socks)
    {
        scriptengine->Update(frameSeconds);
    }

    for (size_t i = 0; i < socks->size(); i++)
    {
        if ((*socks)[i]->is_open() == false)
        {
            socks->erase(socks->begin() + i);
            i--;
        }
    }
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
    for (auto const& sock : *socks)
    {
        send<unsigned int>(*sock, {MessageKill});
        sock->close();
    }
}

void server(std::string kernelFile, std::vector<std::string> clients)
{
    boost::asio::io_service service;
    socks = std::make_shared<decltype(socks)::element_type>(); // I like C++
    std::ifstream inputFile(kernelFile);
    if (inputFile.is_open() == false)
        throw std::runtime_error("Kernel file \""+kernelFile+"\" didn't exist");
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
        auto sock = std::make_shared<tcp::socket>(service);
        int port = 23456;
        auto colon = arg.find(":");
        if (colon != std::string::npos)
        {
            port = std::stoi(arg.substr(colon + 1, arg.length() - colon - 1));
            arg = arg.substr(0, colon);
        }
        sock->connect(tcp::endpoint(boost::asio::ip::address::from_string(arg), port));
        send<unsigned int>(*sock, {MessageKernelSource});

        writeStr(*sock, sourcecode);

        read<uint8_t>(*sock, 1);

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
