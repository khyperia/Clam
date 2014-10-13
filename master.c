#define _POSIX_C_SOURCE 200112L
#include <time.h>
#undef _POSIX_C_SOURCE
#include "master.h"
#include <GL/glut.h>
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include "luaHelper.h"
#include "socketHelper.h"
#include "helper.h"

int* sockets = NULL;
bool keyboard[UCHAR_MAX + 1] = {false};
int animationFrame = 0;
lua_State* luaState = NULL;
struct timespec lastFrame;
double fps = 0.0;

void keyboardDownFunc(unsigned char c, int UNUSED x, int UNUSED y)
{
    keyboard[c] = true;
}

void keyboardUpFunc(unsigned char c, int UNUSED x, int UNUSED y)
{
    keyboard[c] = false;
}

void masterIdleFunc(void)
{
    struct timespec newFrame;
    if (PrintErr(clock_gettime(CLOCK_MONOTONIC, &newFrame)))
    {
        puts("Exiting.");
        exit(-1);
    }
    double elapsed = (double)(newFrame.tv_sec - lastFrame.tv_sec) +
                    (double)(newFrame.tv_nsec - lastFrame.tv_nsec) / 1000000000;
    fps = (1 / elapsed + fps * 3) / 4;
    if (PrintErr(runLua(luaState, elapsed)))
    {
        puts("Exiting.");
        exit(-1);
    }
    lastFrame = newFrame;
    glutPostRedisplay();
}

void masterDisplayFunc(void)
{
    float animationFrameVal = (float)fmod(animationFrame / 1000.0f, 1.0f);
    if (animationFrameVal > 0.5)
        animationFrameVal = 1 - animationFrameVal;
    animationFrameVal *= 2;
    if (animationFrame < 1000)
        glClearColor(animationFrameVal, 0, 0, 1);
    else if (animationFrame < 2000)
        glClearColor(0, animationFrameVal, 0, 1);
    else
        glClearColor(0, 0, animationFrameVal, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    animationFrame = (animationFrame + 1) % 3000;

    char buffer[100];
    snprintf(buffer, 100, "FPS: %f", fps);
    glRasterPos2i(-1, 1);
    for (int i = 0; buffer[i]; i++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, buffer[i]);

    glutSwapBuffers();

    if (PrintErr((int)glGetError()))
    {
        puts("Exiting.");
        exit(-1);
    }
}

void doOnExit_master(void)
{
    if (sockets)
    {
        int* sock = sockets;
        while (*sock)
        {
            if (*sock != -1)
            {
                int msgTerm = MessageTerm;
                send_p(*sock, &msgTerm, sizeof(int));
                close(*sock);
            }
            sock++;
        }
        free(sockets);
    }
    if (luaState)
        deleteLua(luaState);
}

int* connectToSlaves(char* slaves)
{
    size_t numIps;
    char* slavesDup = my_strdup(slaves);
    for (numIps = 0; strtok(numIps == 0 ? slavesDup : NULL, "~"); numIps++)
    {
    }
    free(slavesDup);

    int* ips = malloc_s(numIps * sizeof(int*) + 1);
    for (size_t i = 0; i < numIps; i++)
    {
        char* slaveIp = strtok(i == 0 ? slaves : NULL, "~");
        if (!slaveIp)
        {
            puts("Internal error, strtok returned null when it shouldn't have");
            free(ips);
            return NULL;
        }
        char* slavePort = strchr(slaveIp, ':');
        if (slavePort)
        {
            *slavePort = '\0';
            slavePort++;
        }
        else
        {
            slavePort = "23456";
        }
        printf("Connecting to %s:%s\n", slaveIp, slavePort);
        ips[i] = connectSocket(slaveIp, slavePort);
        if (ips[i] <= 0)
        {
            printf("Unable to connect to %s:%s.\n", slaveIp, slavePort);
            free(ips);
            return NULL;
        }
    }
    ips[numIps] = 0;
    return ips;
}

int main(int argc, char** argv)
{
    if (PrintErr(atexit(doOnExit_master)))
        puts("Continuing anyway");

    if (argc != 2)
    {
        printf("Usage: %s [lua filename]\n", argv[0]);
        return -1;
    }

    char* ips = my_strdup(sgetenv("CLAM2_SLAVES", "127.0.0.1:23456"));
    sockets = connectToSlaves(ips);
    free(ips);
    if (!sockets)
    {
        puts("Exiting.");
        return -1;
    }

    if (PrintErr(newLua(&luaState, argv[1])))
    {
        puts("Exiting.");
        return -1;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutCreateWindow("Clam2 Master");

    if (PrintErr(clock_gettime(CLOCK_MONOTONIC, &lastFrame)))
    {
        puts("Exiting.");
        return -1;
    }

    glutKeyboardFunc(keyboardDownFunc);
    glutKeyboardUpFunc(keyboardUpFunc);
    glutIdleFunc(masterIdleFunc);
    glutDisplayFunc(masterDisplayFunc);
    glutMainLoop();
    puts("glutMainLoop returned, that's odd");
    return -1;
}
