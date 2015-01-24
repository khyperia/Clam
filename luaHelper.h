#pragma once

#include "helper.h"
#include <lua.h>

// Note: `state` must be a local variable of type lua_State* to use this macro
#define LuaPrintErr(expr) (PrintErr(expr) ? luaL_error(state, "LuaPrintErr assertion fail") : 0)

int newLua(lua_State** state, const char* filename, char** args);

void deleteLua(lua_State* state);

int runLua(lua_State* state, double frameSeconds);
