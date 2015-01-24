extern "C" {
#include <lua.h>
#include <lauxlib.h>
}

#include <vrpn_Tracker.h>

static vrpn_Tracker_Remote* remote = NULL;

double pos[3];
double look[3];
double up[3];

// http://answers.unity3d.com/questions/372371
static void mul_quat_vec(double* result, const double quat[4], const double vec[3])
{
    const double num = quat[0] * 2;
    const double num2 = quat[1] * 2;
    const double num3 = quat[2] * 2;
    const double num4 = quat[0] * num;
    const double num5 = quat[1] * num2;
    const double num6 = quat[2] * num3;
    const double num7 = quat[0] * num2;
    const double num8 = quat[0] * num3;
    const double num9 = quat[1] * num3;
    const double num10 = quat[3] * num;
    const double num11 = quat[3] * num2;
    const double num12 = quat[3] * num3;
    result[0] = (1 - (num5 + num6)) * vec[0] + (num7 - num12) * vec[1] + (num8 + num11) * vec[2];
    result[1] = (num7 + num12) * vec[0] + (1 - (num4 + num6)) * vec[1] + (num9 - num10) * vec[2];
    result[2] = (num8 - num11) * vec[0] + (num9 + num10) * vec[1] + (1 - (num4 + num5)) * vec[2];
}

static void VRPN_CALLBACK handle_tracker(void*, const vrpn_TRACKERCB t)
{
    for (int i = 0; i < 3; i++)
        pos[i] = t.pos[i];
    double templook[] = {1, 0, 0};
    mul_quat_vec(look, t.quat, templook);
    double tempup[] = {0, 1, 0};
    mul_quat_vec(up, t.quat, tempup);
}

void setfield(lua_State* luaState, int index, double value)
{
    lua_pushnumber(luaState, index);
    lua_pushnumber(luaState, value);
    lua_settable(luaState, -3);
}

void pushvec(lua_State* luaState, double vals[3])
{
    lua_newtable(luaState);
    for (int i = 0; i < 3; i++)
        setfield(luaState, i + 1, vals[i]);
}

int run_vrpn(lua_State* luaState)
{
    if (!remote)
    {
        const char* remoteAddr = luaL_checkstring(luaState, 1);
        remote = new vrpn_Tracker_Remote(remoteAddr);
        remote->register_change_handler(NULL, handle_tracker);
    }
    remote->mainloop();
    pushvec(luaState, pos);
    pushvec(luaState, look);
    pushvec(luaState, up);
    return 3;
}

extern "C" {
int luaopen_vrpn_help(lua_State* luaState)
{
    lua_register(luaState, "vrpn", run_vrpn);
    return 0;
}
}
