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

// Check that arguments are correct.
// `types` contains LUA_T* variables, *with an extra LUA_TNONE to signify the end*
int checkArgs(lua_State* state, int* types)
{
    int argc = lua_gettop(state);
    int argIdx = 1;
    while (types[argIdx - 1] != LUA_TNONE)
    {
        if (argIdx - 1 >= argc)
        {
            puts("Too many arguments");
            return -1;
        }
        if (lua_type(state, argIdx) != types[argIdx - 1])
        {
            printf("Argument %d was wrong: `%s` is param, `%s` is required\n", argIdx,
                    lua_typename(state, lua_type(state, argIdx)),
                    lua_typename(state, types[argIdx - 1]));
            return -1;
        }
        argIdx++;
    }
    if (argIdx - 1 != argc)
    {
        puts("Not enough arguments");
        return -1;
    }
    return 0;
}

// Note: `state` must be a local variable of type lua_State* to use this macro
#define LuaPrintErr(expr) if (PrintErr(expr)) luaL_error(state, "LuaPrintErr assertion fail")

// Checks if the character passed in is a currently pressed key
int run_iskeydown(lua_State* state)
{
    int types[] = { LUA_TSTRING, LUA_TNONE };
    LuaPrintErr(checkArgs(state, types));
    const char* str = lua_tostring(state, 1);
    LuaPrintErr(strlen(str) != 1);
    lua_pushboolean(state, keyboard[(unsigned char)str[0]]);
    return 1;
}

// Unsets a key so iskeydown() does not return true the next time(s) it is called
// (until the key is pressed again)
int run_unsetkey(lua_State* state)
{
    int types[] = { LUA_TSTRING, LUA_TNONE };
    LuaPrintErr(checkArgs(state, types));
    const char* str = lua_tostring(state, 1);
    LuaPrintErr(strlen(str) != 1);
    keyboard[(unsigned char)str[0]] = false;
    return 0;
}

// Sends data to all sockets;
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

// Sends an enum MessageType to all sockets
int send_all_msg(enum MessageType messageType)
{
    return send_all(&messageType, sizeof(messageType));
}

// Sends a string to all sockets
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

// Receives an int from all sockets and checks if it is zero
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

// Creates a buffer with arg(1) id, and arg(2) size
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

// Deletes the buffer with arg(1) id
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

// Downloads a buffer to a png on disk
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
    LuaPrintErr(recv_all());
    return 0;
}

// Uploads a png to a buffer
int run_uplbuffer(lua_State* state)
{
    int types[] = { LUA_TNUMBER, LUA_TSTRING, LUA_TNONE };
    LuaPrintErr(checkArgs(state, types));
    LuaPrintErr(send_all_msg(MessageUplBuffer));
    int bufferId = (int)lua_tointeger(state, 1);
    const char* filename = lua_tostring(state, 2);
    long width = 0, height = 0;
    float* buffer = loadPng(filename, &width, &height);
    if (!buffer)
    {
        LuaPrintErr("loadPng error" && 1);
    }
    long size = width * height * 4 * 4;
    LuaPrintErr(send_all(&bufferId, sizeof(bufferId)));
    LuaPrintErr(send_all(&size, sizeof(size)));
    LuaPrintErr(send_all(buffer, (size_t)size));
    LuaPrintErr(recv_all());
    lua_pushnumber(state, width);
    lua_pushnumber(state, height);
    return 2;
}

// Compiles a kernel with the sources in the files specified by the args
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

// Invokes a kernel
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

// Reloads the current lua file (simply re-executes the current file)
int run_reload(lua_State* state)
{
    lua_getglobal(state, "__file__");
    const char* str = lua_tostring(state, -1);
    printf("Reloading %s\n", str);
    LuaPrintErr(luaL_dofile(state, str));
    lua_pop(state, 1);
    return 0;
}

// Creates a lua state, setting globals and such
int newLua(lua_State** state, const char* filename, char** args)
{
    *state = luaL_newstate();
    luaL_openlibs(*state);
    lua_pushstring(*state, filename);
    lua_setglobal(*state, "__file__");
    lua_register(*state, "iskeydown", run_iskeydown);
    lua_register(*state, "unsetkey", run_unsetkey);
    lua_register(*state, "dlbuffer", run_dlbuffer);
    lua_register(*state, "uplbuffer", run_uplbuffer);
    lua_register(*state, "mkbuffer", run_mkbuffer);
    lua_register(*state, "rmbuffer", run_rmbuffer);
    lua_register(*state, "compile", run_compile);
    lua_register(*state, "kernel", run_kernel);
    lua_register(*state, "reload", run_reload);
    lua_newtable(*state);
    for (int i = 1; *args; i++, args++)
    {
        lua_pushnumber(*state, i);
        lua_pushstring(*state, *args);
        lua_settable(*state, -3);
    }
    lua_setglobal(*state, "arg");

    if (PrintErr(luaL_dofile(*state, filename)))
    {
        printf("Lua failed to parse file:\n%s\n", lua_tostring(*state, -1));
        lua_pop(*state, 1);
        return -1;
    }
    return 0;
}

// Deletes a lua state
void deleteLua(lua_State* state)
{
    lua_close(state);
}

// Calls the update() function of a lua state
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
