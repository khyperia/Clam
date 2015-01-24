#include "helper.h"
#include "socketHelper.h"
#include "glclContext.h"
#include "slaveSocket.h"

SDL_Window* window = NULL;
SDL_GLContext glContext = NULL;
TCPsocket socketFd;
struct ScreenPos screenPos;
struct Interop interop;

int updateSlave(void)
{
    if (PrintErr(slaveSocket(&interop, socketFd, screenPos)))
        return -1;

    int newWidth, newHeight;
    SDL_GetWindowSize(window, &newWidth, &newHeight);
    if (newWidth != screenPos.width || newHeight != screenPos.height)
    {
        screenPos.width = newWidth;
        screenPos.height = newHeight;
        if (screenPos.width <= 0 || screenPos.height <= 0)
        {
            puts("Window width or height was zero. This causes bad problems.");
            return -1;
        }
        if (PrintErr(resizeInterop(&interop, screenPos.width, screenPos.height)))
        {
            return -1;
        }
    }
    if (PrintErr(blitInterop(interop, screenPos.width, screenPos.height)))
    {
        return -1;
    }
    SDL_GL_SwapWindow(window);
    return 0;
}

void doOnExit_slave(void)
{
    if (socketFd)
        SDLNet_TCP_Close(socketFd);
    if (window)
        SDL_DestroyWindow(window);
    if (glContext)
        SDL_GL_DeleteContext(glContext);
    deleteInterop(interop);
    puts("Exiting");
}

int eventFilter(void UNUSED* userData, SDL_Event* event)
{
    return event->type == SDL_QUIT;
}

int main(int UNUSED argc, char UNUSED** argv)
{
    if (PrintErr(atexit(doOnExit_slave)))
        puts("Continuing anyway");
    if (PrintErr(SDLNet_Init()))
        exit(EXIT_FAILURE);
    const char* portS = sgetenv("CLAM2_PORT", "23456");
    int port;
    if (PrintErr(sscanf(portS, "%d", &port) != 1))
    {
        puts("Bad CLAM2_PORT");
        exit(EXIT_FAILURE);
    }
    TCPsocket hostsocket = hostSocket((Uint16)port);
    if (!hostsocket)
    {
        printf("Failed to host socket: %s\n", SDLNet_GetError());
        exit(EXIT_FAILURE);
    }

    {
        puts("Waiting for connection");
        SDLNet_SocketSet set = SDLNet_AllocSocketSet(1);
        SDLNet_TCP_AddSocket(set, hostsocket);
        if (PrintErr(SDLNet_CheckSockets(set, 100000) != 1))
        {
            puts("Accepting socket timed out");
            SDLNet_FreeSocketSet(set);
            exit(EXIT_FAILURE);
        }
        SDLNet_FreeSocketSet(set);
    }

    puts("Accepting connection");
    socketFd = SDLNet_TCP_Accept(hostsocket);
    SDLNet_TCP_Close(hostsocket);
    if (!socketFd)
    {
        printf("Failed to accept socket: %s\n", SDLNet_GetError());
        perror("accept");
        exit(EXIT_FAILURE);
    }

    const char* winpos = sgetenv("CLAM2_WINPOS", "1024x1024+0+0");
    int winX, winY;
    if (sscanf(winpos, "%dx%d%d%d", &screenPos.width, &screenPos.height, &winX, &winY) != 4)
    {
        puts("CLAM2_WINPOS not in correct format (WIDTHxHEIGHT+OFFX+OFFY).");
        exit(EXIT_FAILURE);
    }

    const char* renderPos = sgetenv("CLAM2_RENDERPOS", "-512,-512");
    if (sscanf(renderPos, "%d,%d", &screenPos.x, &screenPos.y) != 2)
    {
        puts("CLAM2_RENDERPOS not in correct format ([+-]OFFX,[+-]OFFY).");
        exit(EXIT_FAILURE);
    }

    int isFullscreen = !strcmp(sgetenv("CLAM2_FULLSCREEN", "false"), "true");

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
    SDL_SetEventFilter(eventFilter, NULL);

    Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
    if (isFullscreen)
        flags |= SDL_WINDOW_FULLSCREEN;
    window = SDL_CreateWindow("Clam2 Slave", winX, winY, screenPos.width, screenPos.height, flags);
    glContext = SDL_GL_CreateContext(window);

    if (PrintErr(newInterop(&interop)))
    {
        exit(EXIT_FAILURE);
    }
    if (PrintErr(resizeInterop(&interop, screenPos.width, screenPos.height)))
    {
        exit(EXIT_FAILURE);
    }

    while (1)
    {
        SDL_Event event;
        int isexit = 0;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
                isexit = 1;
        }
        if (isexit)
            break;
        if (PrintErr(updateSlave()))
            exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
