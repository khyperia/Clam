#pragma once

#include <string>
#include <vector>

void echo(const char* incomingPort, std::vector<std::string> outgoingIps, bool keepProcessAlive)
    __attribute__((noreturn));
