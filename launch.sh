#!/bin/bash

if [ -z $1 ]
then
    echo "Usage: $0 [kernel name]"
    exit 1
fi

export PROGRAMTYPE=slave
export SLAVE_PORT=23456
export SLAVE_WINDOW=1000x1000+0+0
export SLAVE_FULLSCREEN=false
./bin/clam2 &

sleep 1;

export PROGRAMTYPE=master
export MASTER_KERNEL=$1
export MASTER_CONNECTIONS=127.0.0.1:23456
./bin/clam2
