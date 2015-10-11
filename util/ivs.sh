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

ssh -t -t ${IVS_USER}@${IVS_HOSTNAME} "cd ${IVS_TEMP_DIR}/util; ./ivs_middle.sh \"${CLAM3_KERNEL}\" \"${IVS_TEMP_DIR}\" \"${HOST_PORT}\""

printMessage "SSH exited, waiting for host to exit"

wait

printMessage "Exit success"
