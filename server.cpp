#include "server.h"
#include "interop.h"
#include <memory>
#include <set>
#include <fstream>
#include <chrono>
#include <GL/glut.h>
#include "socket.h"

using boost::asio::ip::tcp;
std::vector<std::shared_ptr<tcp::socket>> socks;
float offsetX = 0.f, offsetY = 0.f, zoom = 1.f;
std::set<unsigned char> pressedKeys;
std::set<int> pressedSpecialKeys;
auto lastUpdate = std::chrono::steady_clock::now();

void idleFuncServer()
{
    auto thisUpdate = std::chrono::steady_clock::now();
    auto frameSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(thisUpdate - lastUpdate).count() / 1000.f;
    lastUpdate = thisUpdate;

    if (pressedSpecialKeys.find(GLUT_KEY_UP) != pressedSpecialKeys.end())
        offsetY -= zoom * frameSeconds;
    if (pressedSpecialKeys.find(GLUT_KEY_DOWN) != pressedSpecialKeys.end())
        offsetY += zoom * frameSeconds;
    if (pressedSpecialKeys.find(GLUT_KEY_LEFT) != pressedSpecialKeys.end())
        offsetX -= zoom * frameSeconds;
    if (pressedSpecialKeys.find(GLUT_KEY_RIGHT) != pressedSpecialKeys.end())
        offsetX += zoom * frameSeconds;
    if (pressedKeys.find('w') != pressedKeys.end())
        zoom /= 1 + frameSeconds;
    if (pressedKeys.find('s') != pressedKeys.end())
        zoom *= 1 + frameSeconds;

    for (auto const& sock : socks)
    {
        try
        {
            send<unsigned int>(*sock, {1});
            writeStr(*sock, "main"); // kernel name

            send<int>(*sock, {-1});
            writeStr(*sock, "");

            send<int>(*sock, {-2});

            send<int>(*sock, { sizeof(float) });
            send<float>(*sock, { offsetX });

            send<int>(*sock, { sizeof(float) });
            send<float>(*sock, { offsetY });

            send<int>(*sock, { sizeof(float) });
            send<float>(*sock, { zoom });

            send<unsigned int>(*sock, {0});
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock->close();
        }
    }

    for (auto const& sock : socks)
    {
        try
        {
            if (sock->is_open())
                read<unsigned int>(*sock, 1);
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock->close();
        }
    }

    for (size_t i = 0; i < socks.size(); i++)
    {
        if (socks[i]->is_open() == false)
        {
            socks.erase(socks.begin() + i);
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
    for (auto const& sock : socks)
    {
        send<unsigned int>(*sock, {UINT_MAX});
        sock->close();
    }
}

void server(int argc, char** argv)
{
    glutInit(&argc, argv);

    boost::asio::io_service service;
    argv++;
    while (*argv)
    {
        auto sock = std::make_shared<tcp::socket>(service);
        std::string arg = *argv;
        int port = 23456;
        auto colon = arg.find(":");
        if (colon != std::string::npos)
        {
            port = std::stoi(arg.substr(colon + 1, arg.length() - colon - 1));
            arg = arg.substr(0, colon);
        }
        sock->connect(tcp::endpoint(boost::asio::ip::address::from_string(arg), port));
        send<unsigned int>(*sock, {2});
        writeStr(*sock, "");

        std::ifstream inputFile("kernel.cl");
        std::string sourcecode((std::istreambuf_iterator<char>(inputFile)),
                std::istreambuf_iterator<char>());

        writeStr(*sock, sourcecode);

        read<unsigned int>(*sock, 1);

        socks.push_back(sock);
        argv++;
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
