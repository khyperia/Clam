#pragma once

#include "vector.h"

class vrpn_Tracker_Remote;

class VrpnHelp
{
    vrpn_Tracker_Remote *con;
public:
    Vector3<double> pos;
    Vector3<double> look;
    Vector3<double> up;

    VrpnHelp();

    ~VrpnHelp();

    void MainLoop();
};
