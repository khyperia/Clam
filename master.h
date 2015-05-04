#pragma once

#include "socketHelper.h"
#include <limits.h>
#include <stdbool.h>

extern bool keyboard[UCHAR_MAX + 1];

bool pumpEvents();
extern TCPsocket* sockets;
