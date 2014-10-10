#include "luaHelper.h"
#include "helper.h"
#include "socketHelper.h"
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

void assertArgs(lua_State* state, int* types)
{
    if (checkArgs(state, types))
    {
        lua_pushstring(state, "Argument types for function were incorrect");
        lua_error(state);
    }
}

int run_iskeydown(lua_State* state)
{
    int types[] = { LUA_TSTRING, LUA_TNONE };
    assertArgs(state, types);
    const char* str = lua_tostring(state, 1);
    if (strlen(str) != 1)
    {
        lua_pushstring(state, "key functions must take single-character strings");
        lua_error(state);
    }
    lua_pushboolean(state, keyboard[(unsigned char)str[0]]);
    return 1;
}

int run_unsetkey(lua_State* state)
{
    int types[] = { LUA_TSTRING, LUA_TNONE };
    assertArgs(state, types);
    const char* str = lua_tostring(state, 1);
    if (strlen(str) != 1)
    {
        lua_pushstring(state, "key functions must take single-character strings");
        lua_error(state);
    }
    keyboard[(unsigned char)str[0]] = false;
    return 0;
}

int send_all(void* data, int numBytes)
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

int run_dlbuffer(lua_State* state)
{
    int types[] = { LUA_TNUMBER, LUA_TNONE };
    assertArgs(state, types);
    lua_pushstring(state, "dlbuffer not implemented");
    lua_error(state);
    return 0;
}

int run_mkbuffer(lua_State* state)
{
    int types[] = { LUA_TNUMBER, LUA_TNUMBER, LUA_TNONE };
    assertArgs(state, types);
    lua_pushstring(state, "mkbuffer not implemented");
    lua_error(state);
    return 0;
}

int run_rmbuffer(lua_State* state)
{
    int types[] = { LUA_TNUMBER, LUA_TNONE };
    assertArgs(state, types);
    lua_pushstring(state, "mkbuffer not implemented");
    lua_error(state);
    return 0;
}

int run_compile(lua_State* state)
{
    send_all_msg(MessageKernelSource);
    int argc = lua_gettop(state);
    if (PrintErr(send_all(&argc, sizeof(int))))
    {
        lua_pushstring(state, "compile(): send() failure");
        lua_error(state);
    }
    for (int i = 1; i <= argc; i++)
    {
        if (lua_isstring(state, i) == false)
        {
            lua_pushstring(state, "compile() must be called with only filenames");
            lua_error(state);
        }
        char* file = readWholeFile(lua_tostring(state, i));
        if (!file)
        {
            lua_pushstring(state, "compile() file read failed");
            lua_error(state);
        }

        if (PrintErr(send_all_str(file)))
        {
            lua_pushstring(state, "compile(): send() failure");
            lua_error(state);
        }

        free(file);
    }
    
    if (PrintErr(recv_all()))
    {
        lua_pushstring(state, "compile(): recv() failure");
        lua_error(state);
    }
    return 0;
}

int run_kernel(lua_State* state)
{
    send_all_msg(MessageKernelInvoke);
    int argc = lua_gettop(state);
    if (argc < 3)
    {
        lua_pushstring(state, "kernel(): not enough arguments");
        lua_error(state);
    }
    if (!lua_isstring(state, 1) || !lua_isnumber(state, 2) || !lua_isnumber(state, 3))
    {
        lua_pushstring(state, "kernel(): wrong types of first three arguments");
        lua_error(state);
    }
    const char* kernelName = lua_tostring(state, 1);
    long launchSize[2] = { lua_tointeger(state, 2), lua_tointeger(state, 3) };
    if (PrintErr(send_all_str(kernelName)))
    {
        lua_pushstring(state, "kernel(): send() failure");
        lua_error(state);
    }
    if (PrintErr(send_all(launchSize, sizeof(launchSize))))
    {
        lua_pushstring(state, "kernel(): send() failure");
        lua_error(state);
    }
    for (int i = 4; i <= argc; i++)
    {
        switch (lua_type(state, i))
        {
            case LUA_TFUNCTION:
                {
                    int screenSizeMagicVal = -2;
                    if (PrintErr(send_all(&screenSizeMagicVal, sizeof(int))))
                    {
                        lua_pushstring(state, "kernel(): send() failure");
                        lua_error(state);
                    }
                }
                break;
            case LUA_TSTRING:
                {
                    if (!lua_isnumber(state, i))
                    {
                        lua_pushstring(state,
                                "kernel(): buffer names must be a string representing a number");
                        lua_error(state);
                    }
                    int value[2] = { -1, 0 };
                    value[1] = lua_tointeger(state, i);
                    if (PrintErr(send_all(&value, sizeof(value))))
                    {
                        lua_pushstring(state, "kernel(): send() failure");
                        lua_error(state);
                    }
                }
                break;
            case LUA_TNUMBER:
                {
                    int size = sizeof(float);
                    float value = lua_tonumber(state, i);
                    if (PrintErr(send_all(&size, sizeof(size))) ||
                        PrintErr(send_all(&value, size)))
                    {
                        lua_pushstring(state, "kernel(): send() failure");
                        lua_error(state);
                    }
                }
                break;
            case LUA_TTABLE:
                {
                    lua_pushinteger(state, 1);
                    lua_gettable(state, i);
                    int value[2] = { sizeof(int), lua_tointeger(state, -1) };
                    lua_pop(state, 1);
                    if (PrintErr(send_all(&value, sizeof(value))))
                    {
                        lua_pushstring(state, "kernel(): send() failure");
                        lua_error(state);
                    }
                }
                break;
            default:
                {
                    lua_pushstring(state, "Unknown argument type in kernel()");
                    lua_error(state);
                }
                break;
        }
    }
    
    int doneValue = 0;
    if (PrintErr(send_all(&doneValue, sizeof(int))))
    {
        lua_pushstring(state, "kernel: send() failure");
        lua_error(state);
    }

    if (PrintErr(recv_all()))
    {
        lua_pushstring(state, "compile(): recv() failure");
        lua_error(state);
    }
    return 0;
}

int newLua(lua_State** state, const char* filename)
{
    *state = luaL_newstate();
    luaL_openlibs(*state);
    lua_register(*state, "iskeydown", run_iskeydown);
    lua_register(*state, "unsetkey", run_unsetkey);
    lua_register(*state, "dlbuffer", run_dlbuffer);
    lua_register(*state, "mkbuffer", run_mkbuffer);
    lua_register(*state, "rmbuffer", run_rmbuffer);
    lua_register(*state, "compile", run_compile);
    lua_register(*state, "kernel", run_kernel);

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
