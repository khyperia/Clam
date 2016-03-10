#pragma once

#include <string>
#include <vector>

void ParseCmdline(int argc, char **argv);

std::string MasterIp();

std::string HostBindPort();

bool IsCompute();

bool IsUserInput();

std::string WindowPos();

std::string KernelName();

int Headless(int *numTimes);

bool RenderOffset(int *shiftx, int *shifty);

std::string VrpnName();

std::string FontName();

std::vector<int> CudaDeviceNums();

bool DoSaveProgress();
