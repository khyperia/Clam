#!/usr/bin/env bash

PRELUDE="
set -o errexit
set -o nounset
"
${PRELUDE}

SOURCE_LOC=".."
IVS_USER="${USER}"
IVS_HOSTNAME="ivs.research.mtu.edu"
IVS_TEMP_DIR="/research/${IVS_USER}/temp-clam3"
SLAVE_IPS=("tile-0-0" "tile-0-1" "tile-0-2" "tile-0-3" "tile-0-4" "tile-0-5" "tile-0-6" "tile-0-7")
SLAVE_PORT=11235
HOST_PORT=23457
SCREENWIDTH=5760
SCREENHEIGHT=1080
MAX_SCREENCOORDS_X=2
MAX_SCREENCOORDS_Y=4
CUDA_TOOLKIT_ROOT_DIR=/opt/cuda
export CLAM3_KERNEL=mandelbox

# Prints a very readable bold message that stands out
function printMessage()
{
    echo "$(tput sgr 0)$(tput bold)== ${BASH_SOURCE} ===> $(tput setaf 1)$@$(tput sgr 0)"
}

printMessage "Copying sources to ${IVS_HOSTNAME}"

rsync -ah -e ssh --exclude=.git --exclude=build -c --partial --inplace --delete ${SOURCE_LOC} ${IVS_USER}@${IVS_HOSTNAME}:${IVS_TEMP_DIR}

printMessage "Launching local client"

CLAM3_HOST=${HOST_PORT} ./clam3 &

sleep 1

printMessage "SSHing to ${IVS_HOSTNAME}"

RUN_ON_TILE="
${PRELUDE}
cd \"${IVS_TEMP_DIR}/build\"
case \${SLAVE} in
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
        echo \"Unknown tile location for \${SLAVE}\"
        ;;
esac
renderposX=\$(bc <<< \"$SCREENWIDTH * (\${WIN_X} - $MAX_SCREENCOORDS_X / 2))
renderposY=\$(bc <<< \"$SCREENHEIGHT * (\${WIN_Y} - $MAX_SCREENCOORDS_Y / 2))
export CLAM3_RENDEROFFSET=\${renderposX}x\${renderposY}
export CLAM3_KERNEL=${CLAM3_KERNEL}
export CLAM3_CONNECT=\${ORIGINAL_HOST}:${HOST_PORT}
export CLAM3_WINDOWPOS=${SCREENWIDTH}x${SCREENHEIGHT}+0+0
export DISPLAY=:0
./clam3
"

ssh -t -t ${IVS_USER}@${IVS_HOSTNAME} "
${PRELUDE}
cd \"${IVS_TEMP_DIR}\"
mkdir -p build
cd build
echo \" --- Running CMake\"
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
echo \" --- Building\"
make
echo \" --- Connecting to clients...\"
ORIGINAL_HOST=\`cut -d' ' -f1 <<< \"\${SSH_CLIENT}\"\`

for SLAVE in ${SLAVE_IPS[*]}
do
  echo \" --- Connecting to \${SLAVE}\"
  ssh -p ${SLAVE_PORT} -t -t \${SLAVE} \"${RUN_ON_TILE}\" &
  sleep 1
done

wait
"

printMessage "SSH exited, waiting for host to exit"

wait

printMessage "Exit success"
