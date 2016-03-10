#pragma once

#include <string>

void ParseCmdline(int argc, char **argv);

std::string MasterIp();

std::string HostBindPort();

bool IsCompute();

bool IsUserInput();

std::string WindowPos();

std::string KernelName();

std::string GpuName();

int Headless(int *numTimes);

bool RenderOffset(int *shiftx, int *shifty);

std::string VrpnName();

std::string FontName();

int CudaDeviceNum();

bool DoSaveProgress();
