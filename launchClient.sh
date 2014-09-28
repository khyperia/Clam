#!/bin/bash

if [ -z $1 ]
then
    echo "Usage: $0 [kernel name]"
    exit 1
fi

export PROGRAMTYPE=master
export MASTER_KERNEL=$1
export MASTER_CONNECTIONS=127.0.0.1:23456~127.0.0.1:23456
./bin/clam2
