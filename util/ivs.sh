#!/usr/bin/env bash

set -o errexit
set -o nounset

SOURCE_LOC=".."
IVS_USER="${IVS_USER:-$USER}"
IVS_HOSTNAME="ivs.research.mtu.edu"
IVS_TEMP_DIR="/research/${IVS_USER}/temp-clam3"
HOST_PORT=23457
CUDA_TOOLKIT_ROOT_DIR=/opt/cuda
export CLAM3_KERNEL=mandelbox
if [ -e $HOME/.vrpn-server ]
then
    export CLAM3_VRPN=$(<$HOME/.vrpn-server)
fi
if [ -e /usr/share/fonts/dejavu/DejaVuLGCSansMono.ttf ]
then
    export CLAM3_FONT=/usr/share/fonts/dejavu/DejaVuLGCSansMono.ttf
fi

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

ssh -R ${HOST_PORT}:localhost:${HOST_PORT} -t -t ${IVS_USER}@${IVS_HOSTNAME} "cd ${IVS_TEMP_DIR}/util; ./ivs_middle.sh \"${CLAM3_KERNEL}\" \"${IVS_TEMP_DIR}\" \"${HOST_PORT}\""

printMessage "SSH exited, waiting for host to exit"

wait

printMessage "Exit success"
