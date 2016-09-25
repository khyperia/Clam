#!/usr/bin/env bash

set -o errexit
set -o nounset

CLAM3_KERNEL=$1
IVS_TEMP_DIR=$2
HOST_PORT=$3
SLAVE_IPS=("tile-0-0" "tile-0-1" "tile-0-2" "tile-0-3" "tile-0-4" "tile-0-5" "tile-0-6" "tile-0-7")
CUDA_TOOLKIT_ROOT_DIR=/share/apps/cuda

# Prints a very readable bold message that stands out
function printMessage()
{
    echo "== ${BASH_SOURCE} ===> $@"
}

cd "${IVS_TEMP_DIR}"
mkdir -p build
cd build

printMessage "Running CMake"
# Because we're running as an ssh script (no login), .bashrc isn't sourced,
# so our $PATH doesn't get set up to include wherever cmake is on IVS.
# It also sets the CUDA50 variable, so use that instead of our preset (if it exists)
test -e /share/apps/bin/bashrc && . /share/apps/bin/bashrc # The relevant script where PATH is added to
CUDA_TOOLKIT_ROOT_DIR=${CUDA50:-${CUDA_TOOLKIT_ROOT_DIR}}
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}

printMessage "Building"
make

printMessage "Connecting to clients..."
for SLAVE in "${SLAVE_IPS[@]}"
do
  printMessage "Connecting to ${SLAVE}"
  ssh -R ${HOST_PORT}:localhost:${HOST_PORT} ${SLAVE} "cd ${IVS_TEMP_DIR}/build; ./ivs_tile.sh \"${CLAM3_KERNEL}\" \"${IVS_TEMP_DIR}\" \"localhost:${HOST_PORT}\" 0" &
  # only first ssh needs reverse portforward
  ssh ${SLAVE} "cd ${IVS_TEMP_DIR}/build; ./ivs_tile.sh \"${CLAM3_KERNEL}\" \"${IVS_TEMP_DIR}\" \"localhost:${HOST_PORT}\" 1" &
  sleep 0.1
done

printMessage "Waiting for slaves"
wait

printMessage "IVS script done"
