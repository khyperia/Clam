#pragma once
#include <lua.h>

int newLua(lua_State** state, const char* filename, char** args);
void deleteLua(lua_State* state);
int runLua(lua_State* state, double frameSeconds);
