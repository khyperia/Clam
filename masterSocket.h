#pragma once

#include "socketHelper.h"

extern int numWaitingSoftSync;

int masterSocketRecv(TCPsocket* socketFds);
