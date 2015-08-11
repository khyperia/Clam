#pragma once

#include <string>

void ParseCmdline(int argc, char **argv);

std::string MasterIp();

std::string HostBindPort();

bool IsCompute();

bool IsUserInput();

std::string WindowPos();

std::string KernelName();

int Headless();

bool RenderOffset(int *shiftx, int *shifty);