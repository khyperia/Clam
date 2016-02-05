#!/usr/bin/env bash

set -o errexit
set -o nounset

# Prints a very readable bold message that stands out
function printMessage()
{
    echo "$(tput sgr 0)$(tput bold)== ${BASH_SOURCE} ===> $(tput setaf 1)$@$(tput sgr 0)"
}

SOURCE_LOC=".."
IVS_USER="${IVS_USER:-$USER}"
IVS_HOSTNAME="ivs.research.mtu.edu"
IVS_TEMP_DIR="/research/${IVS_USER}/temp-clam3"
HOST_PORT=23457
VRPN_OBJECT=Wand
FONT_ATTEMPT=/usr/share/fonts/TTF/DejaVuSansMono.ttf
export CLAM3_KERNEL=mandelbox
if [ -e $HOME/.vrpn-server ]
then
    export CLAM3_VRPN=${VRPN_OBJECT}@$(<$HOME/.vrpn-server)
fi
if [ -e ${FONT_ATTEMPT} ]
then
    export CLAM3_FONT=${FONT_ATTEMPT}
else
    printMessage "${FONT_ATTEMPT} not found, will probably crash. Fix ivs.sh FONT_ATTEMPT to point at a ttf font."
fi

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
