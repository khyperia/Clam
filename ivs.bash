#!/bin/bash
IVS_USER="$USER"
IVS_HOSTNAME="127.0.0.1"
IVS_TEMP_DIR="/home/${IVS_USER}/temp-clam2"
ECHO_LISTEN_PORT=23456
# Ports required in IP address
ECHO_CONNECT_TO="127.0.0.1:23448 127.0.0.1:23449 127.0.0.1:23450 127.0.0.1:23451 127.0.0.1:23452 127.0.0.1:23453 127.0.0.1:23454 127.0.0.1:23455"
SCREENWIDTH=500
SCREENHEIGHT=300

OFFSET_WINDOWS=0
MAX_SCREENCOORDS_X=2
MAX_SCREENCOORDS_Y=4
function getScreenCoords() {
    case $1 in
        127.0.0.1:23448)
            echo 0 0
            ;;
        127.0.0.1:23449)
            echo 0 1
            ;;
        127.0.0.1:23450)
            echo 0 2
            ;;
        127.0.0.1:23451)
            echo 0 3
            ;;
        127.0.0.1:23452)
            echo 1 0
            ;;
        127.0.0.1:23453)
            echo 1 1
            ;;
        127.0.0.1:23454)
            echo 1 2
            ;;
        127.0.0.1:23455)
            echo 1 3
            ;;
        *)
            echo 0 0
            return 1
            ;;
    esac
}

ECHO_CONNECT_TO_TEMP_ARR=(${ECHO_CONNECT_TO})
CLAM2_SLAVES=$(printf "~%s" $(printf "${IVS_HOSTNAME}:${ECHO_LISTEN_PORT} %.0s" $(seq ${#ECHO_CONNECT_TO_TEMP_ARR[@]})))
export CLAM2_SLAVES=${CLAM2_SLAVES:1}
ECHO_ARGS="$ECHO_LISTEN_PORT ${ECHO_CONNECT_TO//:/ }"

trap 'cleanup' ERR
trap 'cleanup' INT
cleanupNoexit() {
    rm ./.temp-clam2-ssh-socket
    kill -TERM `jobs -p` &> /dev/null
}
cleanup() {
    echo 'Cleaning up'
    cleanupNoexit
    exit 1
}

# Prints a very readable bold message that stands out
function printMessage()
{
    echo "$(tput sgr 0)$(tput bold)== $BASH_SOURCE ===> $(tput setaf 1)$@$(tput sgr 0)"
}

# Make sure our current directory is the same place as where this script is.
SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ "${SCRIPT_DIRECTORY}" != "${PWD}" ]]; then
    echo "This script is in '${SCRIPT_DIRECTORY}'."
    echo "You are currently in '${PWD}'."
    echo
    echo "Use 'cd $SCRIPT_DIRECTORY' to change into the same directory as this script and try again."
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "Usage:"
    echo "$0 script/[scriptname].lua [args to lua script]"
    exit 1
fi

printMessage "Running make bin/clam2_master"
make bin/clam2_master

# Create a persistant ssh connection that we will reuse. This will
# just make it so we have to SSH into ivs once (might be slow, might
# prompt for a password) but then subsequent ssh calls are nearly
# instantanious.
#  Use -x to explicitly disable X forwarding since we don't need it (and the user might have specified it as an option in their own ssh config file)
# Use -S to create/specify an ssh control socket.
# Use -t to force tty allocation.
# Use -q to supress warning/diagnostic messages.
# Use -oBatchMode=yes to cause a failure if a password prompt appears.
printMessage "Connecting to IVS..."
rm -rf ./.temp-dgr-ssh-socket
ssh -oBatchMode=yes -q -t -t -x -M -S ./.temp-clam2-ssh-socket ${IVS_USER}@${IVS_HOSTNAME} "sleep 1d" &
sleep 1
if [[ ! -r ./.temp-clam2-ssh-socket ]]; then
	echo "We failed to establish an SSH control socket."
	echo "You can typically resolve this problem by:"
	echo " (1) Creating an ssh key with no password and the default options (run 'ssh-keygen')"
	echo " (2) Copy the contents of ~/.ssh/id_rsa.pub from your computer and paste it into a file named ~/.ssh/authorized_keys on the IVS machine (if the authorized_keys file exists, paste it at the bottom of the file."

	# If we don't exit in this situation, this script would prompt for
	# the password for every SSH call below.
	exit
fi
printMessage "Connected to IVS."

# Create an ssh command with appropriate arguments that we can use
# repeatedly to run programs on IVS. 
SSH_CMD="ssh -q -t -t -x -S ./.temp-clam2-ssh-socket ${IVS_USER}@${IVS_HOSTNAME}"

printMessage "Creating $IVS_TEMP_DIR on IVS"
${SSH_CMD} mkdir -p "$IVS_TEMP_DIR"

printMessage "Copying CWD to $IVS_TEMP_DIR"
rsync -ah -e ssh --exclude=.svn --exclude=.git --exclude=bin --exclude=obj --checksum --partial --no-whole-file --inplace --delete . ${IVS_USER}@${IVS_HOSTNAME}:${IVS_TEMP_DIR}

printMessage "Running sync on IVS"
${SSH_CMD} sync

printMessage "Running make bin/clam2_echo on IVS"
${SSH_CMD} make -C "${IVS_TEMP_DIR}" bin/clam2_echo bin/clam2_slave PKGCONFIGCFLAGS=-I/export/apps/cuda-5.0/include/ LDFLAGS_MASTER=""

printMessage "Running echo server on IVS"
${SSH_CMD} "${IVS_TEMP_DIR}/bin/clam2_echo ${ECHO_ARGS}" &

for slave in $ECHO_CONNECT_TO; do
    xyArr=($(getScreenCoords $slave))
    screenX=$(bc <<< "$OFFSET_WINDOWS * $SCREENWIDTH * ${xyArr[0]}")
    screenY=$(bc <<< "$OFFSET_WINDOWS * $SCREENHEIGHT * ${xyArr[1]}")
    renderposX=$(bc <<< "$SCREENWIDTH * ${xyArr[0]} - $SCREENWIDTH * $MAX_SCREENCOORDS_X / 2")
    renderposY=$(bc <<< "$SCREENHEIGHT * ${xyArr[1]} - $SCREENHEIGHT * $MAX_SCREENCOORDS_Y / 2")
    port=(${slave//:/ })
    tile=${port[0]}
    port=${port[1]}

    RUN_ON_TILE="cd ${IVS_TEMP_DIR}
export CLAM2_WINPOS=${SCREENWIDTH}x${SCREENHEIGHT}+${screenX}+${screenY}
export CLAM2_RENDERPOS=${renderposX},${renderposY}
export CLAM2_PORT=${port}
export DISPLAY=:0.0
./bin/clam2_slave"
    
    printMessage "Running slave slave ${tile} at coordinates ${renderposX},${renderposY}"
    ${SSH_CMD} "ssh -t ${tile} \"${RUN_ON_TILE}\"" &
done

echo "Waiting a second for slaves to boot"
sleep 1

printMessage "Running master on localhost"
./bin/clam2_master $1

# Loop until all jobs are completed. If there is only one job
# remaining, it is probably the ssh master connection. We can kill
# that by deleting the socket.
while (( 1 )); do
    sleep 1
    jobs
    if [[ `jobs | wc -l` -lt 2 ]]; then
	    printMessage "Looks like everything finished successfully, cleaning up..."
	    cleanupNoexit
	    exit 0
    fi
done
