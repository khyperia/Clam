#include "client.h"
#include "interop.h"
#include "helper.h"
#include <memory>
#include <thread>
#include <set>
#include <GL/glut.h>
#include "socket.h"

int oldWidth = -1, oldHeight = -1, oldX = -1, oldY = -1;
int screensaverFrame = 0;
std::string hostPort;
bool resized = false;
std::shared_ptr<ClamContext> context;
std::shared_ptr<ClamInterop> interop;
std::shared_ptr<ClamKernel> kernel;
std::shared_ptr<CSocket> acceptor;
std::shared_ptr<CSocket> sock;

void connectThreaded()
{
    try
    {
        if (acceptor == nullptr)
            acceptor = std::make_shared<CSocket>(hostPort.c_str());
        puts("Waiting for connection");
        sock = acceptor->Accept();
    }
    catch (std::exception const& ex)
    {
        puts("Failed to host server:");
        puts(ex.what());
    }
}

void rebootDisplay()
{
    context = nullptr;
    interop = nullptr;
    kernel = nullptr;
    sock = nullptr;
    std::thread(connectThreaded).detach();
}

void displayFunc()
{
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    if (kernel)
    {
        if (width != oldWidth || height != oldHeight)
        {
            if (width > 0 && height > 0)
                interop->Resize(kernel, width, height);
            else
                puts("width/height was zero, things may fail");
        }
        interop->Blit(*kernel->GetQueue());
        glutSwapBuffers();
    }
    oldWidth = width;
    oldHeight = height;
    if (!sock)
    {
        const int totalScreensaverFrames = 1000;
        screensaverFrame = (screensaverFrame + 1) % totalScreensaverFrames;
        float r, g, b;
        hsvToRgb(static_cast<float>(screensaverFrame) / totalScreensaverFrames,
                0.5, 0.5, r, g, b);
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
    if (sock && !context)
    {
        puts("Connected, starting render client");
        context = std::make_shared<ClamContext>();
        interop = std::make_shared<ClamInterop>(context->GetContext());
    }
    try
    {
        std::vector<unsigned int> messageType;
        while (sock && (messageType = sock->RecvMaybe<unsigned int>(1)).size() != 0)
        {
            switch (messageType[0])
            {
                case MessageNull:
                    break;
                case MessageKernelInvoke:
                    {
                        if (!kernel)
                            throw std::runtime_error("Kernel was not compiled before"
                                    "a MessageKernelInvoke happened");
                        auto kernName = sock->RecvStr();
                        auto launchSize = sock->Recv<int>(2);
                        for (int index = 0;; index++)
                        {
                            int arglen = sock->Recv<int>(1)[0];
                            if (arglen == 0)
                                break;
                            else if (arglen == -1) // buffer name
                            {
                                auto arg = interop->GetBuffer(sock->RecvStr());
                                kernel->SetArg(kernName, static_cast<cl_uint>(index),
                                        sizeof(*arg), arg.get());
                            }
                            else if (arglen == -2) // *FOUR* parameters, x, y, width, height
                            {
                                kernel->SetArg(kernName, static_cast<cl_uint>(index++),
                                        sizeof(int), &oldX);
                                kernel->SetArg(kernName, static_cast<cl_uint>(index++),
                                        sizeof(int), &oldY);
                                kernel->SetArg(kernName, static_cast<cl_uint>(index++),
                                        sizeof(int), &interop->width);
                                kernel->SetArg(kernName, static_cast<cl_uint>(index),
                                        sizeof(int), &interop->height);
                            }
                            else
                            {
                                auto arg = sock->Recv<uint8_t>(arglen); // recieve bytes of arg
                                kernel->SetArg(kernName, static_cast<cl_uint>(index),
                                        static_cast<size_t>(arglen), arg.data());
                            }
                        }
                        if (launchSize[0] == -1)
                            launchSize[0] = oldWidth;
                        if (launchSize[1] == -1)
                            launchSize[1] = oldHeight;
                        kernel->Invoke(kernName,
                                static_cast<unsigned long>(launchSize[0]),
                                static_cast<unsigned long>(launchSize[1]));
                        sock->Send<uint8_t>({0});
                    }
                    break;
                case MessageKernelSource:
                    {
                        std::vector<std::string> sourcecode;
                        while (true)
                        {
                            std::string temp = sock->RecvStr();
                            if (temp.empty())
                                break;
                            sourcecode.push_back(temp);
                        }
                        kernel = std::make_shared<ClamKernel>(context->GetContext(),
                                context->GetDevice(), sourcecode);
                        if (oldWidth > 0 && oldHeight > 0)
                            interop->Resize(kernel, oldWidth, oldHeight);
                        else
                            puts("width/height have not been initialized, things may fail");
                        sock->Send<uint8_t>({0});
                    }
                    break;
                case MessageMkBuffer:
                    {
                        auto buffername = sock->RecvStr();
                        auto size = sock->Recv<long>(1)[0];
                        interop->MkBuffer(buffername, size);
                        sock->Send<uint8_t>({0});
                    }
                    break;
                case MessageDlBuffer:
                    {
                        auto buffername = sock->RecvStr();
                        auto size = sock->Recv<long>(1)[0];
                        interop->DlBuffer(kernel->GetQueue(), buffername, size);
                        sock->Send<uint8_t>({0});
                    }
                    break;
                case MessageRmBuffer:
                    {
                        auto buffername = sock->RecvStr();
                        interop->RmBuffer(buffername);
                        sock->Send<uint8_t>({0});
                    }
                    break;
                case MessageKill:
                    puts("Caught shutdown signal, rebooting");
                    rebootDisplay();
                    break;
                default:
                    throw std::runtime_error("Unknown packet id "
                            + std::to_string(messageType[0]));
            }
        }
    }
    catch (std::exception const& ex)
    {
        puts("Exception in client idleFunc:");
        puts(ex.what());
        sock = nullptr;
        rebootDisplay();
    }
    glutPostRedisplay();
}

void client(const char* clientPort,
        int xpos, int ypos, int width, int height,
        int renderX, int renderY,
        bool fullscreen)
{
    hostPort = clientPort;
    oldX = renderX;
    oldY = renderY;
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowPosition(xpos, ypos);
    glutInitWindowSize(width, height);
    glutCreateWindow("Clam2");
    if (fullscreen)
        glutFullScreen();

    rebootDisplay();
    glutDisplayFunc(displayFunc);
    glutIdleFunc(idleFunc);
    glutMainLoop();
}
