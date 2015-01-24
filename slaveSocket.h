#pragma once

#include "glclContext.h"
#include "socketHelper.h"

struct ScreenPos
{
    int x, y, width, height;
};

int slaveSocket(struct Interop* interop, TCPsocket socketFd, struct ScreenPos screenPos);
