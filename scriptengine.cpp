#include "scriptengine.h"
#include "server.h" // for iskeydown

#include <fstream>
#include <stdexcept>

int run_iskeydown(lua_State* luaState)
{
    int argc = lua_gettop(luaState);
    if (argc != 1)
    {
        lua_pushstring(luaState, "Must be called as iskeydown(char)");
        lua_error(luaState);
    }
    std::string s = lua_tostring(luaState, 1);
    if (s.length() != 1)
    {
        lua_pushstring(luaState, "Must be called as iskeydown(char)");
        lua_error(luaState);
    }
    bool result = iskeydown(s[0]);
    lua_pushboolean(luaState, result);
    return 1;
}

int run_unsetkey(lua_State* luaState)
{
    int argc = lua_gettop(luaState);
    if (argc != 1)
    {
        lua_pushstring(luaState, "Must be called as unsetkey(char)");
        lua_error(luaState);
    }
    std::string s = lua_tostring(luaState, 1);
    if (s.length() != 1)
    {
        lua_pushstring(luaState, "Must be called as iskeydown(char)");
        lua_error(luaState);
    }
    unsetkey(s[0]);
    return 0;
}

int run_mkbuffer(lua_State* luaState)
{
    int argc = lua_gettop(luaState);
    if (argc != 1 && argc != 2)
    {
        lua_pushstring(luaState,
                "Must be called as mkbuffer(string [, sizeInBytes]), one or two args");
        lua_error(luaState);
    }
    std::string bufferName = lua_tostring(luaState, 1);
    if (bufferName.empty())
    {
        lua_pushstring(luaState,
                "Must be called as mkbuffer(string [, sizeInBytes])");
        lua_error(luaState);
    }
    long bufferSize = argc == 2 ? lua_tointeger(luaState, 2) : -1;
    auto socks = getSocks();
    for (auto& sock : *socks)
    {
        try
        {
            sock->Send<unsigned int>({MessageMkBuffer});
            sock->SendStr(bufferName);
            sock->Send<long>({bufferSize});
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    for (auto& sock : *socks)
    {
        try
        {
            if (sock->Recv<uint8_t>(1)[0] != 0)
                throw std::runtime_error("Client failed to make buffer");
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    while (true)
    {
        auto found = std::find(socks->begin(), socks->end(), nullptr);
        if (found == socks->end())
            break;
        socks->erase(found);
    }
    return 0;
}

int run_rmbuffer(lua_State* luaState)
{
    int argc = lua_gettop(luaState);
    if (argc != 1)
    {
        lua_pushstring(luaState,
                "Must be called as rmbuffer(string), one arg");
        lua_error(luaState);
    }
    std::string bufferName = lua_tostring(luaState, 1);
    if (bufferName.empty())
    {
        lua_pushstring(luaState,
                "Must be called as rmbuffer(string)");
        lua_error(luaState);
    }
    auto socks = getSocks();
    for (auto& sock : *socks)
    {
        try
        {
            sock->Send<unsigned int>({MessageRmBuffer});
            sock->SendStr(bufferName);
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    for (auto& sock : *socks)
    {
        try
        {
            if (sock->Recv<uint8_t>(1)[0] != 0)
                throw std::runtime_error("Client failed to remove buffer");
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    while (true)
    {
        auto found = std::find(socks->begin(), socks->end(), nullptr);
        if (found == socks->end())
            break;
        socks->erase(found);
    }
    return 0;
}

int run_dlbuffer(lua_State* luaState)
{
    int argc = lua_gettop(luaState);
    if (argc != 2)
    {
        lua_pushstring(luaState, "Must be called as dlbuffer(string, int), two args");
        lua_error(luaState);
    }
    std::string bufferName = lua_tostring(luaState, 1);
    if (bufferName.empty())
    {
        lua_pushstring(luaState, "Must be called as dlbuffer(string, int)");
        lua_error(luaState);
    }
    long imageWidth = lua_tointeger(luaState, 2);
    auto socks = getSocks();
    for (auto& sock : *socks)
    {
        try
        {
            sock->Send<unsigned int>({MessageDlBuffer});
            sock->SendStr(bufferName);
            sock->Send<long>({imageWidth});
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    for (auto& sock : *socks)
    {
        try
        {
            if (sock->Recv<uint8_t>(1)[0] != 0)
                throw std::runtime_error("Client failed to download buffer");
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    while (true)
    {
        auto found = std::find(socks->begin(), socks->end(), nullptr);
        if (found == socks->end())
            break;
        socks->erase(found);
    }
    return 0;
}

int run_kernel(lua_State* luaState)
{
    int argc = lua_gettop(luaState);

    if (argc < 4)
    {
        lua_pushstring(luaState,
        "kernel must be called as kernel(\"functionname\", launchwidth, launchheight, args...)");
        lua_error(luaState);
    }
    std::string kernelName = lua_tostring(luaState, 1);
    int launchWidth = static_cast<int>(lua_tointeger(luaState, 2));
    int launchHeight = static_cast<int>(lua_tointeger(luaState, 3));
    auto socks = getSocks();

    if (!socks)
        throw std::runtime_error("Socks was null");

    for (auto& sock : *socks)
    {
        try
        {
            sock->Send<unsigned int>({MessageKernelInvoke});
            sock->SendStr(kernelName); // kernel name
            sock->Send<int>({launchWidth, launchHeight});

            for (int arg = 4; arg <= argc; arg++)
            {
                switch (lua_type(luaState, arg))
                {
                    case LUA_TFUNCTION:
                        sock->Send<int>({-2}); // x, y, width, height
                        break;
                    case LUA_TSTRING:
                        sock->Send<int>({-1}); // buffer
                        sock->SendStr(lua_tostring(luaState, arg)); // empty = screen
                        break;
                    case LUA_TNUMBER:
                        sock->Send<int>({ sizeof(float) });
                        sock->Send<float>({ static_cast<float>(lua_tonumber(luaState, arg)) });
                        break;
                    case LUA_TTABLE:
                        {
                            lua_pushinteger(luaState, 1);
                            lua_gettable(luaState, arg);
                            int value = static_cast<int>(lua_tointeger(luaState, -1));
                            lua_pop(luaState, 1);
                            sock->Send<int>({ sizeof(int) });
                            sock->Send<int>({ value });
                        }
                        break;
                    default:
                        throw std::runtime_error("Unknown argument type in kernel() arg index "
                                + std::to_string(arg));
                }
            }

            sock->Send<int>({0}); // end arglist    
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    for (auto& sock : *socks)
    {
        try
        {
            if (sock->Recv<uint8_t>(1)[0] != 0)
                throw std::runtime_error("Client failed to launch kernel");
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    while (true)
    {
        auto found = std::find(socks->begin(), socks->end(), nullptr);
        if (found == socks->end())
            break;
        socks->erase(found);
    }
    return 0; // num return values
}

int run_compile(lua_State* luaState)
{
    int argc = lua_gettop(luaState);
    std::vector<std::string> sourcecode;
    for (int i = 1; i <= argc; i++)
    {
        if (lua_isstring(luaState, i))
        {
            std::string kernelFile = lua_tostring(luaState, i);
            std::ifstream inputFile(kernelFile);
            if (inputFile.is_open() == false)
                throw std::runtime_error("Kernel file \"" + kernelFile + "\" didn't exist");
            std::string newsourcecode((std::istreambuf_iterator<char>(inputFile)),
                std::istreambuf_iterator<char>());
            sourcecode.push_back(newsourcecode);
        }
        else if (lua_istable(luaState, i))
        {
            lua_pushnil(luaState);
            std::string newsource;
            while (lua_next(luaState, i) != 0)
            {
                if (!lua_isstring(luaState, -2))
                {
                    lua_pushstring(luaState, "compile() dicts must only have string keys");
                    lua_error(luaState);
                }
                std::string key = lua_tostring(luaState, -2);
                std::string value = lua_tostring(luaState, -1);
                if (key.find_first_of("\t\r\n ") != std::string::npos)
                {
                    lua_pushstring(luaState, "compile() keys must not have spaces");
                    lua_error(luaState);
                }
                newsource += "#define " + key + " " + value + "\n";
                lua_pop(luaState, 1);
            }
            sourcecode.push_back(newsource);
        }
        else
        {
            lua_pushstring(luaState,
                        ("compile must be called as compile([string|dict], ...) - type was "
                        + std::string(lua_typename(luaState, lua_type(luaState, i)))).c_str());
            lua_error(luaState);
        }
    }
    auto socks = getSocks();

    if (!socks)
        throw std::runtime_error("Socks was null");

    for (auto& sock : *socks)
    {
        try
        {
            sock->Send<unsigned int>({MessageKernelSource});
            for (auto const& str : sourcecode)
                if (str.empty() == false)
                    sock->SendStr(str);
            sock->SendStr("");
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    for (auto& sock : *socks)
    {
        try
        {
            if (sock->Recv<uint8_t>(1)[0] != 0)
                throw std::runtime_error("Client failed to build kernel");
        }
        catch (std::exception const& ex)
        {
            puts("Disconnected from client");
            puts(ex.what());
            sock = nullptr;
        }
    }
    while (true)
    {
        auto found = std::find(socks->begin(), socks->end(), nullptr);
        if (found == socks->end())
            break;
        socks->erase(found);
    }
    return 0; // num return values
}

ScriptEngine::ScriptEngine(std::string sourcecode)
{
    luaState = luaL_newstate();
    luaL_openlibs(luaState);

    lua_register(luaState, "iskeydown", run_iskeydown);
    lua_register(luaState, "unsetkey", run_unsetkey);
    lua_register(luaState, "dlbuffer", run_dlbuffer);
    lua_register(luaState, "mkbuffer", run_mkbuffer);
    lua_register(luaState, "rmbuffer", run_rmbuffer);
    lua_register(luaState, "compile", run_compile);
    lua_register(luaState, "kernel", run_kernel);

    int err = luaL_dostring(luaState, sourcecode.c_str());
    if (err)
    {
        std::string thing = lua_tostring(luaState, -1);
        lua_pop(luaState, 1);
        throw std::runtime_error(std::to_string(err) + ": lua failed to parse file - " + thing);
    }
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
