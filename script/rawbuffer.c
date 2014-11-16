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

int saveRaw(float* data, long dataSize)
{
    const char* home = getenv("HOME");
    if (!home)
    {
        puts("Cannot save image: $HOME not set");
        return -1;
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/clam2_buffer.dat", home);

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

int dlrawbuffer_callback(float* data, long bufferSize)
{
    if (PrintErr(saveRaw(data, bufferSize)))
    {
        puts("Ignoring raw write error.");
    }
    return 0;
}

int run_dlrawbuffer(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageDlBuffer));
    long callback = (long)&dlrawbuffer_callback;
    LuaPrintErr(send_all(sockets, &callback, sizeof(callback)));
    int bufferId = (int)luaL_checkinteger(state, 1);
    LuaPrintErr(send_all(sockets, &bufferId, sizeof(int)));
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
    lua_pushnumber(state, size);
    return 1;
}

int luaopen_rawbuffer(lua_State* state)
{
    lua_register(state, "dlrawbuffer", run_dlrawbuffer);
    lua_register(state, "uplrawbuffer", run_uplrawbuffer);
    return 0;
}
