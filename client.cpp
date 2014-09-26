#include "client.h"
#include "interop.h"
#include "helper.h"
#include <memory>
#include <thread>
#include <set>
#include <GL/glut.h>
#include "socket.h"

using boost::asio::ip::tcp;

int oldWidth = -1, oldHeight = -1, oldX = -1, oldY = -1;
int screensaverFrame = 0;
int hostPort;
bool resized = false;
std::shared_ptr<ClamContext> context;
std::shared_ptr<ClamInterop> interop;
std::shared_ptr<ClamKernel> kernel;
std::shared_ptr<tcp::socket> connection;
std::shared_ptr<boost::asio::io_service> ioService;

void connectThreaded()
{
    tcp::acceptor acceptor(*ioService, tcp::endpoint(tcp::v4(), hostPort));
    auto tempconnection = std::make_shared<tcp::socket>(*ioService);
    puts("Waiting for connection");
    acceptor.accept(*tempconnection);
    connection = tempconnection;
}

void rebootDisplay()
{
    context = nullptr;
    interop = nullptr;
    kernel = nullptr;
    connection = nullptr;
    if (!ioService)
        ioService = std::make_shared<boost::asio::io_service>();

    std::thread(connectThreaded).detach();
}

void displayFunc()
{
    if (kernel)
    {
        int width = glutGet(GLUT_WINDOW_WIDTH);
        int height = glutGet(GLUT_WINDOW_HEIGHT);
        oldX = glutGet(GLUT_WINDOW_X);
        oldY = glutGet(GLUT_WINDOW_Y);
        if (width != oldWidth || height != oldHeight)
        {
            oldWidth = width;
            oldHeight = height;
            interop->Resize(kernel, width, height);
        }
        interop->Blit(*kernel->GetQueue());
        glutSwapBuffers();
    }
    if (!connection)
    {
        const int totalScreensaverFrames = 1000;
        screensaverFrame = (screensaverFrame + 1) % totalScreensaverFrames;
        float r, g, b;
        hsvToRgb((float)screensaverFrame / totalScreensaverFrames, 0.5, 0.5, r, g, b);
        glViewport(0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
        glClearColor(r, g, b, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glFlush();
        HandleErr(glGetError());
        glutSwapBuffers();
    }
}
void idleFunc()
{
    if (!ioService)
        rebootDisplay(); // bootstrap lifecycle
    if (connection && !context)
    {
        puts("Connected, starting render client");
        context = std::make_shared<ClamContext>();
        interop = std::make_shared<ClamInterop>(context->GetContext());
    }
    try
    {
        while (connection && connection->available())
        {
            unsigned int messageType = read<unsigned int>(*connection, 1)[0];
            switch (messageType)
            {
                case MessageNull:
                    break;
                case MessageKernelInvoke:
                    {
                        auto kernName = readStr(*connection);
                        for (int index = 0;; index++)
                        {
                            int arglen = read<int>(*connection, 1)[0];
                            if (arglen == 0)
                                break;
                            else if (arglen == -1) // buffer name
                            {
                                auto arg = interop->GetBuffer(readStr(*connection));
                                kernel->SetArg(kernName, index, sizeof(*arg), arg.get());
                            }
                            else if (arglen == -2) // *FOUR* parameters, x, y, width, height
                            {
                                int realX = oldX - glutGet(GLUT_SCREEN_WIDTH) / 2;
                                int realY = oldY - glutGet(GLUT_SCREEN_HEIGHT) / 2;
                                kernel->SetArg(kernName, index++, sizeof(int), &realX);
                                kernel->SetArg(kernName, index++, sizeof(int), &realY);
                                kernel->SetArg(kernName, index++, sizeof(int), &interop->width);
                                kernel->SetArg(kernName, index, sizeof(int), &interop->height);
                            }
                            else
                            {
                                auto arg = read<uint8_t>(*connection, arglen);
                                kernel->SetArg(kernName, index, arglen, arg.data());
                            }
                        }
                        kernel->Invoke(kernName);
                        send<uint8_t>(*connection, {0});
                    }
                    break;
                case MessageKernelSource:
                    {
                        auto sourcecode = readStr(*connection);
                        kernel = std::make_shared<ClamKernel>(context->GetContext(),
                                context->GetDevice(), sourcecode.c_str());
                        if (oldWidth > 0 && oldHeight > 0)
                            interop->Resize(kernel, oldWidth, oldHeight);
                        send<uint8_t>(*connection, {0});
                    }
                    break;
                case MessageMkBuffer:
                    {
                        auto buffername = readStr(*connection);
                        interop->MkBuffer(buffername);
                        send<uint8_t>(*connection, {0});
                    }
                    break;
                case MessageKill:
                    puts("Caught shutdown signal, rebooting");
                    rebootDisplay();
                    break;
                default:
                    throw std::runtime_error("Unknown packet id " + std::to_string(messageType));
            }
        }
    }
    catch (std::exception const& ex)
    {
        puts(ex.what());
        if (connection->is_open())
            connection->close();
        rebootDisplay();
    }
    glutPostRedisplay();
}

void client(int port)
{
    hostPort = port;
    glutInitDisplayMode(GLUT_DOUBLE);
    glutCreateWindow("Clam2");
    // TODO: Figure out what screen to put ourselves on
    //glutFullScreen();

    glutDisplayFunc(displayFunc);
    glutIdleFunc(idleFunc);
    glutMainLoop();
}
