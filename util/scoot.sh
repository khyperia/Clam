#!/usr/bin/env bash
export CLAM3_HEADLESS=${1}
# 4k resolution
export CLAM3_WINDOWPOS=3840x2160+0+0
# don't wreck scoot's drives (4k renderstate is nearly a gigabyte)
export CLAM3_SAVEPROGRESS=0

for x in 0 1 2 3
do
    export CLAM3_DEVICE=$x
    ./clam3 &
done

wait
