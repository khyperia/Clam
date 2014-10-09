#pragma once
#include "glclContext.h"

struct ScreenPos
{
    int x, y, width, height;
};

int slaveSocket(struct Interop* interop, int socketFd, struct ScreenPos screenPos);
