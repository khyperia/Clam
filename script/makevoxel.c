#include <lua.h>
#include <lauxlib.h>

int run_makevoxel(lua_State* luaState)
{
    const char* filename = luaL_checkstring(luaState, 1);
    printf("Hello world %s\n", filename);
    return 0;
}

int makevoxel(lua_State* luaState)
{
    lua_register(luaState, "makevoxel", run_makevoxel);
    return 0;
}
