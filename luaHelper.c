#include "luaHelper.h"
#include "helper.h"
#include "socketHelper.h"
#include "master.h"
#include "masterSocket.h"
#include <lualib.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <time.h>

// Note: `state` must be a local variable of type lua_State* to use this macro
#define LuaPrintErr(expr) if (PrintErr(expr)) luaL_error(state, "LuaPrintErr assertion fail")

void sync_waitmessage(lua_State* state)
{
    time_t begin = clock();
    while (numWaitingSoftSync > 0)
    {
        LuaPrintErr(masterSocketRecv(sockets));
        if ((clock() - begin) / CLOCKS_PER_SEC > 10)
            luaL_error(state, "softsync() time out, call softsync() more often to make sure "
                    "the slaves stay in sync");
    }
}

void sync_sendmessage(lua_State* state)
{
    if (numWaitingSoftSync > 0)
        luaL_error(state, "Sync failure: did not wait before sending another sync message");
    LuaPrintErr(send_all_msg(sockets, MessageSync));
    for (TCPsocket* socket = sockets; *socket; socket++)
        numWaitingSoftSync++;
}

int run_softsync(lua_State* state)
{
    sync_waitmessage(state);
    sync_sendmessage(state);
    return 0;
}

int run_hardsync(lua_State* state)
{
    sync_sendmessage(state);
    sync_waitmessage(state);
    return 0;
}

// Creates a buffer with arg(1) id, and arg(2) size
int run_mkbuffer(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageMkBuffer));
    int first = (int)luaL_checkinteger(state, 1);
    long second = luaL_checkinteger(state, 2);
    LuaPrintErr(send_all(sockets, &first, sizeof(int)));
    LuaPrintErr(send_all(sockets, &second, sizeof(long)));
    return 0;
}

// Deletes the buffer with arg(1) id
int run_rmbuffer(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageRmBuffer));
    int first = (int)luaL_checkinteger(state, 1);
    LuaPrintErr(send_all(sockets, &first, sizeof(int)));
    return 0;
}

// Compiles a kernel with the sources in the files specified by the args
int run_compile(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageKernelSource));
    int argc = lua_gettop(state);
    LuaPrintErr(send_all(sockets, &argc, sizeof(int)));
    for (int i = 1; i <= argc; i++)
    {
        LuaPrintErr(lua_isstring(state, i) == false);
        char* file = readWholeFile(luaL_checkstring(state, i));
        LuaPrintErr(!file);

        LuaPrintErr(send_all_str(sockets, file));

        free(file);
    }
    return 0;
}

// Invokes a kernel
int run_kernel(lua_State* state)
{
    LuaPrintErr(send_all_msg(sockets, MessageKernelInvoke));
    int argc = lua_gettop(state);
    LuaPrintErr(argc < 3);

    const char* kernelName = luaL_checkstring(state, 1);
    long launchSize[2] = { luaL_checkinteger(state, 2), luaL_checkinteger(state, 3) };
    LuaPrintErr(send_all_str(sockets, kernelName));
    LuaPrintErr(send_all(sockets, launchSize, sizeof(launchSize)));
    for (int i = 4; i <= argc; i++)
    {
        switch (lua_type(state, i))
        {
            case LUA_TFUNCTION:
                {
                    int screenSizeMagicVal = -2;
                    LuaPrintErr(send_all(sockets, &screenSizeMagicVal, sizeof(int)));
                }
                break;
            case LUA_TSTRING:
                {
                    LuaPrintErr(!lua_isnumber(state, i));
                    int value[2] = { -1, 0 };
                    value[1] = (int)lua_tointeger(state, i);
                    LuaPrintErr(send_all(sockets, &value, sizeof(value)));
                }
                break;
            case LUA_TNUMBER:
                {
                    int size = sizeof(float);
                    float value = (float)luaL_checknumber(state, i);
                    LuaPrintErr(send_all(sockets, &size, sizeof(size)));
                    LuaPrintErr(send_all(sockets, &value, size));
                }
                break;
            case LUA_TTABLE:
                {
                    lua_pushinteger(state, 1);
                    lua_gettable(state, i);
                    int value[2] = { sizeof(int), (int)luaL_checkinteger(state, -1) };
                    lua_pop(state, 1);
                    LuaPrintErr(send_all(sockets, &value, sizeof(value)));
                }
                break;
            default:
                luaL_error(state, "Unknown argument type in kernel()");
                break;
        }
    }

    int doneValue = 0;
    LuaPrintErr(send_all(sockets, &doneValue, sizeof(int)));
    return 0;
}

// Reloads the current lua file (simply re-executes the current file)
int run_reload(lua_State* state)
{
    lua_getglobal(state, "__file__");
    const char* str = luaL_checkstring(state, -1);
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
    lua_register(*state, "softsync", run_softsync);
    lua_register(*state, "hardsync", run_hardsync);
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

    if (PrintErr(luaL_dostring(*state,
                    "package.cpath=package.cpath .. \";./script/lib?.so;./script/lib?.dll\"\n"
                    "package.path=package.path .. \";./script/?.lua\"\n")))
    {
        printf("Lua failed to execute script setup:\n%s\n", lua_tostring(*state, -1));
        lua_pop(*state, 1);
        return -1;
    }

    if (PrintErr(luaL_dofile(*state, filename)))
    {
        printf("Lua failed to execute file:\n%s\n", lua_tostring(*state, -1));
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
        printf("Lua update failed:\n%s\n", luaL_checkstring(state, -1));
        lua_pop(state, 1);
        return -1;
    }
    return 0;
}
