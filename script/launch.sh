#!/bin/bash

if [ -z $1 ]
then
    echo "Usage: $0 [kernel name]"
    exit 1
fi

if [ ! -e ./bin/clam2 ]
then
    echo './bin/clam2 not found, make sure this script is executed as "./bin/clam2", that is, cwd != script/ (also that you actually ran make and the executable exists)'
    exit 1
fi

slavecount=0
echocount=0
mastercount=0
for pid in $(pgrep clam2); do
    PROGRAMTYPE=$(awk -v 'RS=\0' -F '=' '$1=="PROGRAMTYPE" {print $2}' /proc/$pid/environ)
    if [ "$PROGRAMTYPE" == "slave" ]; then
        (( slavecount++ ))
        echo "Found an already-running slave"
    elif [ "$PROGRAMTYPE" == "echo" ]; then
        (( echocount++ ))
        echo "Found an already-running echo"
    elif [ "$PROGRAMTYPE" == "master" ]; then
        (( mastercount++ ))
        echo "Found an already-running master"
    fi
done

if [[ "$slavecount" == "0" ]]
then
    export PROGRAMTYPE=slave
    export SLAVE_PORT=23455
    export SLAVE_WINDOW=800x600+400+300
    export SLAVE_RENDERCOORDS=-400-300
    export SLAVE_FULLSCREEN=false
    ./bin/clam2 &

    #export PROGRAMTYPE=slave
    #export SLAVE_PORT=23454
    #export SLAVE_WINDOW=1000x1000+1000+0
    #export SLAVE_RENDERCOORDS=+0-500
    #export SLAVE_FULLSCREEN=false
    #./bin/clam2 &
fi

if [[ "$echocount" == "0" ]]
then
    export PROGRAMTYPE=echo
    export ECHO_PORT=23456
    export ECHO_CONNECTIONS=127.0.0.1:23455
    #~127.0.0.1:23454
    ./bin/clam2 &
fi

sleep 1;

if [[ "$mastercount" == "0" ]]
then
    export PROGRAMTYPE=master
    export MASTER_LUA=$1
    export MASTER_CONNECTIONS=127.0.0.1:23456
    #~127.0.0.1:23456
    ./bin/clam2 &
fi
