#include <SDL2/SDL.h>
#include "master.h"
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include "luaHelper.h"
#include "helper.h"

SDL_Window* window = NULL;
TCPsocket* sockets = NULL;
bool keyboard[UCHAR_MAX + 1] = {false};
int animationFrame = 0;
lua_State* luaState = NULL;
Uint32 lastSdl2Ticks = 0;
double fps = 1.0;
Uint32 second = 0;

// Closes all sockets and deletes the lua state
void doOnExit_master(void)
{
    if (sockets)
    {
        for (TCPsocket* sock = sockets; *sock; sock++)
        {
            int msgTerm = MessageTerm;
            send_p(*sock, &msgTerm, sizeof(int));
            SDLNet_TCP_Close(*sock);
        }
        free(sockets);
    }
    if (luaState)
        deleteLua(luaState);
    if (window)
        SDL_DestroyWindow(window);
    puts("Exiting");
}

// Parses and connects to all slaves
TCPsocket* connectToSlaves(char* slaves)
{
    size_t numIps;
    char* slavesDup = my_strdup(slaves);
    for (numIps = 0; strtok(numIps == 0 ? slavesDup : NULL, "~"); numIps++)
    {
    }
    free(slavesDup);

    TCPsocket* ips = malloc_s(numIps * sizeof(TCPsocket) + 1);
    for (size_t i = 0; i < numIps; i++)
    {
        char* slaveIpFull = strtok(i == 0 ? slaves : NULL, "~");
        if (!slaveIpFull)
        {
            puts("Internal error, strtok returned null when it shouldn't have");
            free(ips);
            return NULL;
        }

        char slaveIp[strlen(slaveIpFull) + 1];
        int slavePort;
        if (sscanf(slaveIpFull, "%[^:]:%d", slaveIp, &slavePort) != 2)
        {
            printf("IP was not in the correct format hostname:port - %s\n", slaveIpFull);
            free(ips);
            return NULL;
        }

        printf("Connecting to %s:%d\n", slaveIp, slavePort);
        ips[i] = connectSocket(slaveIp, (Uint16)slavePort);
        if (!ips[i])
        {
            printf("Unable to connect to %s:%d.\n", slaveIp, slavePort);
            free(ips);
            return NULL;
        }
    }
    ips[numIps] = 0;
    return ips;
}

int eventFilter(void UNUSED *userdata, SDL_Event* event)
{
    return event->type == SDL_QUIT || event->type == SDL_KEYUP || event->type == SDL_KEYDOWN;
}

int main(int argc, char** argv)
{
    if (PrintErr(atexit(doOnExit_master)))
        puts("Continuing anyway");

    if (argc < 2)
    {
        printf("Usage: %s [lua filename]\n", argv[0]);
        return -1;
    }

    // strdup because it will be mutated
    char* ips = my_strdup(sgetenv("CLAM2_SLAVES", "127.0.0.1:23456"));
    sockets = connectToSlaves(ips);
    free(ips);
    if (!sockets)
        exit(EXIT_FAILURE);

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER);

    SDL_SetEventFilter(eventFilter, NULL);

    window = SDL_CreateWindow("Clam2 Master", 100, 100, 500, 500,
            SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_INPUT_FOCUS);

    // Pass on the rest of the args to lua
    if (PrintErr(newLua(&luaState, argv[1], argv + 2)))
        exit(EXIT_FAILURE);

    while (true)
    {
        SDL_Event event;
        bool exit = false;
        while (!exit && SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
                exit = true;
            if (event.type == SDL_KEYUP || event.type == SDL_KEYDOWN)
            {
                SDL_Keycode keycode = event.key.keysym.sym;
                if (keycode >= 0 && keycode <= UCHAR_MAX)
                    keyboard[(unsigned char)keycode] = event.type == SDL_KEYDOWN;
            }
        }
        if (exit)
            break;

        Uint32 current = SDL_GetTicks();
        double elapsed = (current - lastSdl2Ticks) / 1000.0;
        lastSdl2Ticks = current;
        if (PrintErr(runLua(luaState, elapsed)))
            return -1;
        fps = (elapsed + fps * 29) / 30;
        Uint32 newsecond = current / 1000;
        if (newsecond != second)
        {
            second = newsecond;
            printf("FPS: %f\n", 1 / fps);
        }
    }
    return EXIT_SUCCESS;
}
