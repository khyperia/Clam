#include "scriptengine.h"
#include "server.h" // for iskeydown

#include <stdexcept>

int run_iskeydown(lua_State* luaState)
{
    int argc = lua_gettop(luaState);
    if (argc != 1)
    {
        lua_pushstring(luaState, "kernel must be called as iskeydown(char)");
        lua_error(luaState);
    }
    std::string s = lua_tostring(luaState, 1);
    if (s.length() != 1)
    {
        lua_pushstring(luaState, "kernel must be called as iskeydown(char)");
        lua_error(luaState);
    }
    bool result = iskeydown(s[0]);
    lua_pushboolean(luaState, result);
    return 1;
}

int run_kernel(lua_State* luaState)
{
    int argc = lua_gettop(luaState);

    if (argc < 2)
    {
        lua_pushstring(luaState, "kernel must be called as kernel(\"functionname\", args...)");
        lua_error(luaState);
    }
    std::string kernelName = lua_tostring(luaState, 1);
    auto socks = getSocks();

    if (!socks)
        throw std::runtime_error("Socks was null");

    for (auto const& sock : *socks)
    {
        try
        {
            send<unsigned int>(*sock, {MessageKernelInvoke});
            writeStr(*sock, kernelName); // kernel name

            for (int arg = 2; arg <= argc; arg++)
            {
                switch (lua_type(luaState, arg))
                {
                    case LUA_TNIL:
                        send<int>(*sock, {-2}); // x, y, width, height
                        break;
                    case LUA_TSTRING:
                        send<int>(*sock, {-1}); // buffer
                        writeStr(*sock, lua_tostring(luaState, arg)); // empty = screen
                        break;
                    case LUA_TNUMBER:
                        send<int>(*sock, { sizeof(float) });
                        send<float>(*sock, { (float)lua_tonumber(luaState, arg) });
                        break;
                    default:
                        throw std::runtime_error("Unknown argument type in kernel()");
                }
            }

            send<int>(*sock, {0}); // end arglist    
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock->close();
        }
    }

    for (auto const& sock : *socks)
    {
        try
        {
            if (sock->is_open())
            {
                unsigned int errcode = read<uint8_t>(*sock, 1)[0];
                if (errcode != MessageOkay)
                {
                    puts(("Client was not okay: " + std::to_string(errcode)).c_str());
                    send<unsigned int>(*sock, {MessageKill});
                    sock->close();
                }
            }
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock->close();
        }
    }

    return 0; // num return values
}

ScriptEngine::ScriptEngine(std::string sourcecode)
{
    luaState = luaL_newstate();
    luaopen_io(luaState); // provides io.*
    luaopen_base(luaState);
    luaopen_table(luaState);
    luaopen_string(luaState);
    luaopen_math(luaState);

    lua_register(luaState, "iskeydown", run_iskeydown);
    lua_register(luaState, "kernel", run_kernel);

    int err = luaL_dostring(luaState, sourcecode.c_str());
    if (err)
        throw std::runtime_error(std::to_string(err) + ": lua failed to parse file");
}

ScriptEngine::~ScriptEngine()
{
    lua_close(luaState);
}

void ScriptEngine::Update(double frameSeconds)
{
    lua_getglobal(luaState, "update");
    lua_pushnumber(luaState, frameSeconds);
    int error = lua_pcall(luaState, 1, 0, 0);
    if (error)
    {
        std::string thing = lua_tostring(luaState, -1);
        lua_pop(luaState, 1);
        throw std::runtime_error(thing);
    }
}
