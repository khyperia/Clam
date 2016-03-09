#!/usr/bin/env bash
export CLAM3_HEADLESS=${1}
export CLAM3_WINDOWPOS=3840x2160+0+0
for x in 0 1 2 3
do
    export CLAM3_DEVICE=$x
    export CLAM3_IMAGENAME=gpu${x}
    ./clam3 &
done

wait
