// Note: This is a module using builtin functions of the master program. Edit at your own risk.

#include <lua.h>
#include <lauxlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <png.h>
#include "../helper.h"
#include "../socketHelper.h"
#include "../master.h"
#define LuaPrintErr(expr) if (PrintErr(expr)) luaL_error(state, "LuaPrintErr assertion fail")

int saveRaw(int uid1, int uid2, float* data, long dataSize)
{
    const char* home = getenv("HOME");
    if (!home)
    {
        puts("Cannot save image: $HOME not set");
        return -1;
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/fractal_id%d_sock%d.png", home, uid1, uid2);

    FILE* fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("Could not open %s for writing\n", filename);
        return -1;
    }

    fwrite(data, 1, (size_t)dataSize, fp);
    fclose(fp);
    return 0;
}


int run_dlrawbuffer(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageDlBuffer));
    int bufferId = (int)luaL_checkinteger(state, 1);
    LuaPrintErr(send_all(sockets, &bufferId, sizeof(int)));
    //long width = luaL_checkinteger(state, 2);
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

        if (PrintErr(saveRaw(bufferId, *socket, data, bufferSize)))
        {
            puts("Ignoring raw write error.");
        }

        free(data);
    }
    if (retval)
        luaL_error(state, "At least one slave failed to download buffer: %d", retval);
    LuaPrintErr(recv_all(sockets));
    return 0;
}

int run_uplrawbuffer(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageUplBuffer));
    int bufferId = (int)luaL_checkinteger(state, 1);
    const char* filename = luaL_checkstring(state, 2);
    size_t size;
    float* buffer = readWholeBinFile(filename, &size);
    if (!buffer)
    {
        LuaPrintErr("readWholeBinFile error" && 1);
    }
    LuaPrintErr(send_all(sockets, &bufferId, sizeof(bufferId)));
    LuaPrintErr(send_all(sockets, &size, sizeof(size)));
    LuaPrintErr(send_all(sockets, buffer, size));
    free(buffer);
    LuaPrintErr(recv_all(sockets));
    lua_pushnumber(state, size);
    return 2;
}

int rawbuffer(lua_State* state)
{
    lua_register(state, "dlrawbuffer", run_dlrawbuffer);
    lua_register(state, "uplrawbuffer", run_uplrawbuffer);
    return 0;
}
