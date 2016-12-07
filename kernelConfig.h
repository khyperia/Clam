#pragma once

#include "kernelSetting.h"
#include "kernelControl.h"
#include "uiSetting.h"

class KernelConfiguration: public Immobile
{
    const std::string name;

public:
    KernelConfiguration(const std::string &kernelName);
    const unsigned char *KernelData() const;
    unsigned int KernelLength() const;
    SettingCollection Settings() const;
    std::vector<std::unique_ptr<KernelControl>>
    Controls(GpuKernel &kernel) const;
    std::vector<std::unique_ptr<UiSetting>> UiSettings() const;
};
