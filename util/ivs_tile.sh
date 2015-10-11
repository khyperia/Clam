#!/usr/bin/env bash

export CLAM3_KERNEL=$1
IVS_TEMP_DIR=$2
NCAT_PORT=$3
IVS_ADDR=$(cut -d' ' -f1 <<< "${SSH_CLIENT}")
SCREENWIDTH=5760
SCREENHEIGHT=1080
MAX_SCREENCOORDS_X=2
MAX_SCREENCOORDS_Y=4

cd "${IVS_TEMP_DIR}/build"
SLAVE=$(hostname -s)
case ${SLAVE} in
    tile-0-3)
        WIN_X=0
        WIN_Y=0
        ;;
    tile-0-2)
        WIN_X=0
        WIN_Y=1
        ;;
    tile-0-1)
        WIN_X=0
        WIN_Y=2
        ;;
    tile-0-0)
        WIN_X=0
        WIN_Y=3
        ;;
    tile-0-7)
        WIN_X=1
        WIN_Y=0
        ;;
    tile-0-6)
        WIN_X=1
        WIN_Y=1
        ;;
    tile-0-5)
        WIN_X=1
        WIN_Y=2
        ;;
    tile-0-4)
        WIN_X=1
        WIN_Y=3
        ;;
    *)
        echo "Unknown tile location for ${SLAVE}"
        ;;
esac
renderposX=$(bc <<< "$SCREENWIDTH * (${WIN_X} - $MAX_SCREENCOORDS_X / 2)")
renderposY=$(bc <<< "$SCREENHEIGHT * (${WIN_Y} - $MAX_SCREENCOORDS_Y / 2)")
export CLAM3_RENDEROFFSET=${renderposX}x${renderposY}
export CLAM3_CONNECT=${IVS_ADDR}:${NCAT_PORT}
export CLAM3_WINDOWPOS=${SCREENWIDTH}x${SCREENHEIGHT}+0+0
export DISPLAY=:0
echo " --- Booting ${SLAVE} render process: run ${CLAM3_KERNEL} at ${SCREENWIDTH}x${SCREENHEIGHT}+${renderposX}+${renderposY} connect ${CLAM3_CONNECT}"
./clam3
