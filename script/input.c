// Note: This is a module using builtin functions of the master program. Edit at your own risk.

#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include "../helper.h"
#include "../master.h"

#define LuaPrintErr(expr) if (PrintErr(expr)) luaL_error(state, "LuaPrintErr assertion fail")

// Checks if the character passed in is a currently pressed key
int run_iskeydown(lua_State* state)
{
    const char* str = luaL_checkstring(state, 1);
    LuaPrintErr(strlen(str) != 1);
    lua_pushboolean(state, keyboard[(unsigned char)str[0]]);
    return 1;
}

// Unsets a key so iskeydown() does not return true the next time(s) it is called
// (until the key is pressed again)
int run_unsetkey(lua_State* state)
{
    const char* str = luaL_checkstring(state, 1);
    LuaPrintErr(strlen(str) != 1);
    keyboard[(unsigned char)str[0]] = false;
    return 0;
}

int input(lua_State* state)
{
    lua_register(state, "iskeydown", run_iskeydown);
    lua_register(state, "unsetkey", run_unsetkey);
    return 0;
}
