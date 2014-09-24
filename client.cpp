#include "client.h"
#include "interop.h"
#include <memory>
#include <set>
#include <GL/glut.h>
#include "socket.h"

using boost::asio::ip::tcp;

int oldWidth = -1, oldHeight = -1;
std::shared_ptr<ClamContext> context;
std::shared_ptr<ClamInterop> interop;
std::shared_ptr<tcp::socket> connection;

void displayFunc()
{
    if (ClamKernel::namedKernels.find("") != ClamKernel::namedKernels.end())
    {
        int width = glutGet(GLUT_WINDOW_WIDTH);
        int height = glutGet(GLUT_WINDOW_HEIGHT);
        if (width != oldWidth || height != oldHeight)
        {
            oldWidth = width;
            oldHeight = height;
            interop->Resize(*ClamKernel::namedKernels[""]->GetQueue(), width, height);
        }
        interop->Blit(*ClamKernel::namedKernels[""]->GetQueue());
        glutSwapBuffers();
    }
}
void idleFunc()
{
    if (connection->is_open() == false)
    {
        puts("Server stream closed, shutting down");
        exit(0);
    }
    while (connection->available())
    {
        unsigned int messageType = read<unsigned int>(*connection, 1)[0];
        switch (messageType)
        {
            case 0:
                break;
            case 1:
                {
                    auto kernName = readStr(*connection);
                    auto kernel = ClamKernel::namedKernels[kernName];
                    for (int index = 0;; index++)
                    {
                        int arglen = read<int>(*connection, 1)[0];
                        if (arglen == 0)
                            break;
                        else if (arglen == -1) // buffer name
                        {
                            auto arg = interop->GetBuffer(readStr(*connection));
                            kernel->SetArg(index, sizeof(*arg), arg.get());
                        }
                        else if (arglen == -2) // *TWO* parameters, int width, int height
                        {
                            kernel->SetArg(index++, sizeof(int), &interop->width);
                            kernel->SetArg(index, sizeof(int), &interop->height);
                        }
                        else
                        {
                            auto arg = read<uint8_t>(*connection, arglen);
                            kernel->SetArg(index, arglen, arg.data());
                        }
                    }
                    kernel->Invoke();
                    send<uint8_t>(*connection, {0});
                }
                break;
            case 2:
                {
                    auto name = readStr(*connection);
                    auto sourcecode = readStr(*connection);
                    ClamKernel::namedKernels[name] = std::make_shared<ClamKernel>(context->GetContext(),
                            context->GetDevice(), sourcecode.c_str());
                    send<uint8_t>(*connection, {0});
                }
                break;
            default:
                throw std::runtime_error("Unknown packet id " + std::to_string(messageType));
        }
    }
    glutPostRedisplay();
}

void client(int argc, char** argv)
{
    int port = 23456;
    if (argv[1])
    {
        std::string arg(argv[1]);
        port = std::stoi(arg.substr(1, arg.length() - 1));
    }
    boost::asio::io_service service;
    tcp::acceptor acceptor(service, tcp::endpoint(tcp::v4(), port));
    connection = std::make_shared<tcp::socket>(service);
    puts("Waiting for connection");
    acceptor.accept(*connection);
    puts("Connected, starting render client");

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutCreateWindow("Clam2");
    context = std::make_shared<ClamContext>();
    interop = std::make_shared<ClamInterop>(context->GetContext());

    glutDisplayFunc(displayFunc);
    glutIdleFunc(idleFunc);
    glutMainLoop();
}
