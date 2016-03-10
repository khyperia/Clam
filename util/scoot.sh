#!/usr/bin/env bash
export CLAM3_HEADLESS=${1}
# 4k resolution
#export CLAM3_WINDOWPOS=3840x2160+0+0
export CLAM3_WINDOWPOS=1920x1080+0+0
# don't wreck scoot's drives (4k renderstate is nearly a gigabyte)
export CLAM3_SAVEPROGRESS=0

export CLAM3_DEVICE=0,1,2,3

./clam3
