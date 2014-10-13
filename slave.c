#include <GL/glut.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include "helper.h"
#include "socketHelper.h"
#include "glclContext.h"
#include "slaveSocket.h"

int socketFd;
struct ScreenPos screenPos;
struct Interop interop;

void masterIdleFunc(void)
{
    if (PrintErr(slaveSocket(&interop, socketFd, screenPos)))
    {
        puts("Exiting.");
        exit(-1);
    }
    glutPostRedisplay();
}

void masterDisplayFunc(void)
{
    int newWidth = glutGet(GLUT_WINDOW_WIDTH);
    int newHeight = glutGet(GLUT_WINDOW_HEIGHT);
    if (newWidth != screenPos.width || newHeight != screenPos.height)
    {
        screenPos.width = newWidth;
        screenPos.height = newHeight;
        if (screenPos.width <= 0 || screenPos.height <= 0)
        {
            puts("Window width or height was zero. This causes bad problems. Exiting.");
            exit(-1);
        }
        if (PrintErr(resizeInterop(&interop, screenPos.width, screenPos.height)))
        {
            puts("Exiting.");
            exit(-1);
        }
    }
    if (PrintErr(blitInterop(interop, screenPos.width, screenPos.height)))
    {
        puts("Exiting.");
        exit(-1);
    }
    glutSwapBuffers();
    if (PrintErr((int)glGetError()))
    {
        puts("OpenGL failure of displayFunc. Exiting.");
        exit(-1);
    }
}

void doOnExit_slave(void)
{
    if (socketFd > 0)
        close(socketFd);
    deleteInterop(interop);
}

int main(int argc, char** argv)
{
    if (PrintErr(atexit(doOnExit_slave)))
        puts("Continuing anyway");
    const char* port = sgetenv("CLAM2_PORT", "23456");
    int hostsocket = hostSocket(port);

    if (hostsocket == -1)
    {
        puts("Bad host socket. Exiting.");
        return -1;
    }

    puts("Accepting connection");
    socketFd = accept(hostsocket, NULL, NULL);
    close(hostsocket);
    if (socketFd == -1)
    {
        perror("accept()");
        puts("Bad client socket. Exiting.");
        return -1;
    }

    const char* winpos = sgetenv("CLAM2_WINPOS", "1024x1024+0+0");
    int winX, winY;
    if (sscanf(winpos, "%dx%d%d%d", &screenPos.width, &screenPos.height, &winX, &winY) != 4)
    {
        puts("CLAM2_WINPOS not in correct format (WIDTHxHEIGHT+OFFX+OFFY). Exiting.");
        return -1;
    }

    const char* renderpos = sgetenv("CLAM2_RENDERPOS", "-512-512");
    if (sscanf(renderpos, "%d%d", &screenPos.x, &screenPos.y) != 2)
    {
        puts("CLAM2_RENDERPOS not in correct format ([+-]OFFX[+-]OFFY). Exiting.");
        return -1;
    }

    glutInit(&argc, argv);
    glutInitWindowSize(screenPos.width, screenPos.height);
    glutInitWindowPosition(winX, winY);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutCreateWindow("Clam2 Slave");

    if (PrintErr(newInterop(&interop)))
    {
        puts("Exiting.");
        return -1;
    }
    if (PrintErr(resizeInterop(&interop, screenPos.width, screenPos.height)))
    {
        puts("Exiting.");
        return -1;
    }

    glutIdleFunc(masterIdleFunc);
    glutDisplayFunc(masterDisplayFunc);
    glutMainLoop();
    puts("glutMainLoop returned, that's odd");
    return -1;
}
