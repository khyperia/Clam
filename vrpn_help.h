#pragma once

#include "vector.h"
#include "movable.h"
#include <string>

#if HAVE_VRPN

#include <vrpn_Tracker.h>

#else
typedef int vrpn_Tracker_Remote;
#endif

class VrpnHelp : public Immobile
{
    vrpn_Tracker_Remote con;
public:
    Vector3<double> pos;
    Vector3<double> look;
    Vector3<double> up;
    VrpnHelp(const std::string &serverName);
    ~VrpnHelp();
    void MainLoop();
};
