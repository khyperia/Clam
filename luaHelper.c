#include "luaHelper.h"
#include "helper.h"
#include "socketHelper.h"
#include "pngHelper.h"
#include "master.h"
#include <lualib.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int checkArgs(lua_State* state, int* types)
{
    int argc = lua_gettop(state);
    int argIdx = 1;
    while (types[argIdx] != LUA_TNONE)
    {
        if (argIdx >= argc)
            return -1;
        if (lua_type(state, argIdx) != types[argIdx])
            return -1;
        argIdx++;
    }
    return 0;
}

// Note: `state` must be a local variable of type lua_State* to use this macro
#define LuaPrintErr(expr) if (PrintErr(expr)) luaL_error(state, "LuaPrintErr assertion fail")

/*
void assertArgs(lua_State* state, int* types)
{
    if (checkArgs(state, types))
    {
        lua_pushstring(state, "Argument types for function were incorrect");
        lua_error(state);
    }
}
*/

int run_iskeydown(lua_State* state)
{
    int types[] = { LUA_TSTRING, LUA_TNONE };
    LuaPrintErr(checkArgs(state, types));
    const char* str = lua_tostring(state, 1);
    LuaPrintErr(strlen(str) != 1);
    lua_pushboolean(state, keyboard[(unsigned char)str[0]]);
    return 1;
}

int run_unsetkey(lua_State* state)
{
    int types[] = { LUA_TSTRING, LUA_TNONE };
    LuaPrintErr(checkArgs(state, types));
    const char* str = lua_tostring(state, 1);
    LuaPrintErr(strlen(str) != 1);
    keyboard[(unsigned char)str[0]] = false;
    return 0;
}

int send_all(void* data, size_t numBytes)
{
    int* socket = sockets;
    int errcode = 0;
    while (*socket)
    {
        if (*socket == -1)
            continue;
        int err = PrintErr(send_p(*socket, data, numBytes));
        if (err)
        {
            close(*socket);
            *socket = -1;
            errcode = err;
        }
        socket++;
    }
    return errcode;
}

int send_all_msg(enum MessageType messageType)
{
    return send_all(&messageType, sizeof(messageType));
}

int send_all_str(const char* string)
{
    int* socket = sockets;
    int errcode = 0;
    while (*socket)
    {
        if (*socket == -1)
            continue;
        int err = PrintErr(send_str(*socket, string));
        if (err)
        {
            close(*socket);
            *socket = -1;
            errcode = err;
        }
        socket++;
    }
    return errcode;
}

int recv_all(void)
{
    int* socket = sockets;
    int errcode = 0;
    while (*socket)
    {
        if (*socket == -1)
            continue;
        int result = 0;
        int err = PrintErr(recv_p(*socket, &result, sizeof(int)));
        if (err)
        {
            close(*socket);
            *socket = -1;
            errcode = err;
        }
        else if (result)
        {
            PrintErr(result && "Slave didn't respond with zero");
            close(*socket);
            *socket = -1;
            errcode = result;
        }
        socket++;
    }
    return errcode;
}

int run_mkbuffer(lua_State* state)
{
    int types[] = { LUA_TNUMBER, LUA_TNUMBER, LUA_TNONE };
    LuaPrintErr(checkArgs(state, types));
    LuaPrintErr(send_all_msg(MessageMkBuffer));
    int first = (int)lua_tointeger(state, 1);
    long second = lua_tointeger(state, 2);
    LuaPrintErr(send_all(&first, sizeof(int)));
    LuaPrintErr(send_all(&second, sizeof(long)));
    LuaPrintErr(recv_all());
    return 0;
}

int run_rmbuffer(lua_State* state)
{
    int types[] = { LUA_TNUMBER, LUA_TNONE };
    LuaPrintErr(checkArgs(state, types));
    LuaPrintErr(send_all_msg(MessageRmBuffer));
    int first = (int)lua_tointeger(state, 1);
    LuaPrintErr(send_all(&first, sizeof(int)));
    LuaPrintErr(recv_all());
    return 0;
}

int run_dlbuffer(lua_State* state)
{
    int types[] = { LUA_TNUMBER, LUA_TNUMBER, LUA_TNONE };
    LuaPrintErr(checkArgs(state, types));
    LuaPrintErr(send_all_msg(MessageDlBuffer));
    int bufferId = (int)lua_tointeger(state, 1);
    LuaPrintErr(send_all(&bufferId, sizeof(int)));
    long width = lua_tointeger(state, 2);
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

        float* data = malloc((size_t)bufferSize);
        if (!data)
        {
            puts("malloc() failed in dlbuffer");
            exit(-1);
        }
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
    LuaPrintErr(recv_all());
    return 0;
}

int run_compile(lua_State* state)
{
    LuaPrintErr(send_all_msg(MessageKernelSource));
    int argc = lua_gettop(state);
    LuaPrintErr(send_all(&argc, sizeof(int)));
    for (int i = 1; i <= argc; i++)
    {
        LuaPrintErr(lua_isstring(state, i) == false);
        char* file = readWholeFile(lua_tostring(state, i));
        LuaPrintErr(!file);

        LuaPrintErr(send_all_str(file));

        free(file);
    }

    LuaPrintErr(recv_all());
    return 0;
}

int run_kernel(lua_State* state)
{
    LuaPrintErr(send_all_msg(MessageKernelInvoke));
    int argc = lua_gettop(state);
    LuaPrintErr(argc < 3);
    LuaPrintErr(!lua_isstring(state, 1) || !lua_isnumber(state, 2) || !lua_isnumber(state, 3));

    const char* kernelName = lua_tostring(state, 1);
    long launchSize[2] = { lua_tointeger(state, 2), lua_tointeger(state, 3) };
    LuaPrintErr(send_all_str(kernelName));
    LuaPrintErr(send_all(launchSize, sizeof(launchSize)));
    for (int i = 4; i <= argc; i++)
    {
        switch (lua_type(state, i))
        {
            case LUA_TFUNCTION:
                {
                    int screenSizeMagicVal = -2;
                    LuaPrintErr(send_all(&screenSizeMagicVal, sizeof(int)));
                }
                break;
            case LUA_TSTRING:
                {
                    LuaPrintErr(!lua_isnumber(state, i));
                    int value[2] = { -1, 0 };
                    value[1] = (int)lua_tointeger(state, i);
                    LuaPrintErr(send_all(&value, sizeof(value)));
                }
                break;
            case LUA_TNUMBER:
                {
                    int size = sizeof(float);
                    float value = (float)lua_tonumber(state, i);
                    LuaPrintErr(send_all(&size, sizeof(size)));
                    LuaPrintErr(send_all(&value, (size_t)size));
                }
                break;
            case LUA_TTABLE:
                {
                    lua_pushinteger(state, 1);
                    lua_gettable(state, i);
                    int value[2] = { sizeof(int), (int)lua_tointeger(state, -1) };
                    lua_pop(state, 1);
                    LuaPrintErr(send_all(&value, sizeof(value)));
                }
                break;
            default:
                luaL_error(state, "Unknown argument type in kernel()");
                break;
        }
    }

    int doneValue = 0;
    LuaPrintErr(send_all(&doneValue, sizeof(int)));

    LuaPrintErr(recv_all());
    return 0;
}

int run_reload(lua_State* state)
{
    lua_getglobal(state, "__file__");
    const char* str = lua_tostring(state, -1);
    printf("Reloading %s\n", str);
    LuaPrintErr(luaL_dofile(state, str));
    lua_pop(state, 1);
    return 0;
}

int newLua(lua_State** state, const char* filename)
{
    *state = luaL_newstate();
    luaL_openlibs(*state);
    lua_pushstring(*state, filename);
    lua_setglobal(*state, "__file__");
    lua_register(*state, "iskeydown", run_iskeydown);
    lua_register(*state, "unsetkey", run_unsetkey);
    lua_register(*state, "dlbuffer", run_dlbuffer);
    lua_register(*state, "mkbuffer", run_mkbuffer);
    lua_register(*state, "rmbuffer", run_rmbuffer);
    lua_register(*state, "compile", run_compile);
    lua_register(*state, "kernel", run_kernel);
    lua_register(*state, "reload", run_reload);

    if (PrintErr(luaL_dofile(*state, filename)))
    {
        printf("Lua failed to parse file:\n%s\n", lua_tostring(*state, -1));
        lua_pop(*state, 1);
        return -1;
    }
    return 0;
}

void deleteLua(lua_State* state)
{
    lua_close(state);
}

int runLua(lua_State* state, double frameSeconds)
{
    lua_getglobal(state, "update");
    lua_pushnumber(state, frameSeconds);
    int error = PrintErr(lua_pcall(state, 1, 0, 0));
    if (error)
    {
        printf("Lua update failed:\n%s\n", lua_tostring(state, -1));
        lua_pop(state, 1);
        return -1;
    }
    return 0;
}
