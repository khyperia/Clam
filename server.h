#pragma once
#include <string>
#include <vector>
#include <memory>
#include "socket.h"

void server(std::string kernelFile, std::vector<std::string> clients);
bool iskeydown(unsigned char key);
void unsetkey(unsigned char key);
std::shared_ptr<std::vector<std::shared_ptr<CSocket>>> getSocks();
