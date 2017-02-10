#include "vrpn_help.h"

#if HAVE_VRPN

// http://answers.unity3d.com/questions/372371
static Vector3<double> mul_quat_vec(const double quat[4], const Vector3<double> &vec)
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
    Vector3<double> result;
    result.x = (1 - (num5 + num6)) * vec.x + (num7 - num12) * vec.y + (num8 + num11) * vec.z;
    result.y = (num7 + num12) * vec.x + (1 - (num4 + num6)) * vec.y + (num9 - num10) * vec.z;
    result.z = (num8 - num11) * vec.x + (num9 + num10) * vec.y + (1 - (num4 + num5)) * vec.z;
    return result;
}

void VRPN_CALLBACK HandleTracker(void *_this_untyped, const vrpn_TRACKERCB info)
{
    VrpnHelp *_this = (VrpnHelp *)_this_untyped;
    _this->pos.x = info.pos[0];
    _this->pos.y = info.pos[1];
    _this->pos.z = info.pos[2];
    _this->look = mul_quat_vec(info.quat, Vector3<double>(1, 0, 0));
    _this->up = mul_quat_vec(info.quat, Vector3<double>(0, 1, 0));
    _this->pos.y = -_this->pos.y;
    _this->look.y = -_this->look.y;
    _this->up.y = -_this->up.y;
}

VrpnHelp::VrpnHelp(const std::string &serverName) :
    con(vrpn_Tracker_Remote(serverName.c_str())), pos(0, 0, 0), look(1, 0, 0), up(0, 1, 0)
{
    con.register_change_handler(this, HandleTracker);
}

VrpnHelp::~VrpnHelp()
{
}

void VrpnHelp::MainLoop()
{
    con.mainloop();
}

#else

#include <iostream>

VrpnHelp::VrpnHelp(const std::string &serverName)
    : con(0), pos(0, 0, 0), look(1, 0, 0), up(0, 1, 0)
{
}

VrpnHelp::~VrpnHelp()
{
}

void VrpnHelp::MainLoop()
{
    static bool printed = false;
    if (!printed)
    {
        printed = true;
        std::cout << "VRPN is disabled, GetVRPN will have no effect" << std::endl;
    }
}

#endif
