#pragma once

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include "socket.h"
#include <string>

class ScriptEngine
{
    lua_State* luaState;
    public:
    ScriptEngine(std::string sourcecode);
    ~ScriptEngine();
    void Update(double frameSeconds);
};
