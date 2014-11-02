#include <lua.h>
#include <lauxlib.h>
#include <unistd.h>
#include <stdlib.h>
#include "../helper.h"
#include "../socketHelper.h"
#include "../master.h"
#include "../pngHelper.h"

#define LuaPrintErr(expr) if (PrintErr(expr)) luaL_error(state, "LuaPrintErr assertion fail")

// Downloads a buffer to a png on disk
int run_dlbuffer(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageDlBuffer));
    int bufferId = (int)luaL_checkinteger(state, 1);
    LuaPrintErr(send_all(sockets, &bufferId, sizeof(int)));
    long width = luaL_checkinteger(state, 2);
    int retval = 0;
    for (int* socket = sockets; *socket; socket++)
    {
        if (*socket == -1)
            continue;
        long bufferSize = 0;
        if (PrintErr(recv_p(*socket, &bufferSize, sizeof(long))))
        {
            close(*socket);
            *socket = -1;
            retval = -1;
            continue;
        }

        if (PrintErr(bufferSize < 0))
        {
            printf("bufferSize = %ld\n", bufferSize);
            close(*socket);
            *socket = -1;
            retval = -1;
            continue;
        }

        float* data = malloc_s((size_t)bufferSize);
        if (PrintErr(recv_p(*socket, data, (size_t)bufferSize)))
        {
            free(data);
            close(*socket);
            *socket = -1;
            retval = -1;
            continue;
        }

        if (PrintErr(savePng(bufferId, *socket, data, width,
                        bufferSize / (width * 4 * (long)sizeof(float)))))
        {
            puts("Ignoring PNG error.");
        }

        free(data);
    }
    if (retval)
        luaL_error(state, "At least one slave failed to download buffer: %d", retval);
    LuaPrintErr(recv_all(sockets));
    return 0;
}

// Uploads a png to a buffer
int run_uplbuffer(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageUplBuffer));
    int bufferId = (int)luaL_checkinteger(state, 1);
    const char* filename = luaL_checkstring(state, 2);
    long width = 0, height = 0;
    float* buffer = loadPng(filename, &width, &height);
    if (!buffer)
    {
        LuaPrintErr("loadPng error" && 1);
    }
    long size = width * height * 4 * 4;
    LuaPrintErr(send_all(sockets, &bufferId, sizeof(bufferId)));
    LuaPrintErr(send_all(sockets, &size, sizeof(size)));
    LuaPrintErr(send_all(sockets, buffer, (size_t)size));
    LuaPrintErr(recv_all(sockets));
    lua_pushnumber(state, width);
    lua_pushnumber(state, height);
    return 2;
}

int imagebuffer(lua_State* state)
{
    lua_register(state, "dlbuffer", run_dlbuffer);
    lua_register(state, "uplbuffer", run_uplbuffer);
    return 0;
}
