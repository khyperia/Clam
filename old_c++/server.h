#pragma once
#include <string>
#include <vector>
#include <memory>
#include "socket.h"

void server(std::string luaFile, std::vector<std::string> clients);
bool iskeydown(char key);
void unsetkey(char key);
std::shared_ptr<std::vector<std::shared_ptr<CSocket>>> getSocks();
