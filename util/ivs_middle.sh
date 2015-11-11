#!/usr/bin/env bash

set -o errexit
set -o nounset

CLAM3_KERNEL=$1
IVS_TEMP_DIR=$2
HOST_PORT=$3
SLAVE_IPS=("tile-0-0" "tile-0-1" "tile-0-2" "tile-0-3" "tile-0-4" "tile-0-5" "tile-0-6" "tile-0-7")
CUDA_TOOLKIT_ROOT_DIR=/export/apps/cuda

cd "${IVS_TEMP_DIR}"
mkdir -p build
cd build

echo " --- Running CMake"
# for some reason [i.e. too lazy to figure it out], .bashrc isn't sourced,
# so our $PATH doesn't get set up to include wherever cmake is on IVS.
if [ -e /share/apps/bin/bashrc ] # The relevant script where PATH is added to
then
    . /share/apps/bin/bashrc
fi
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}

echo " --- Building"
make

echo " --- Connecting to clients..."
ORIGINAL_HOST=$(cut -d' ' -f1 <<< "${SSH_CLIENT}")
for SLAVE in ${SLAVE_IPS[*]}
do
  echo " --- Connecting to ${SLAVE}"
  ssh -t -t ${SLAVE} "cd ${IVS_TEMP_DIR}/build; ./ivs_tile.sh \"${CLAM3_KERNEL}\" \"${IVS_TEMP_DIR}\" \"${ORIGINAL_HOST}:${HOST_PORT}\"" &
  sleep 1
done

echo " --- Waiting for slaves"
wait

echo " --- IVS script done"
