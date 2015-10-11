#!/usr/bin/env bash

CLAM3_KERNEL=$1
IVS_TEMP_DIR=$2
HOST_PORT=$3
SLAVE_IPS=("tile-0-0" "tile-0-1" "tile-0-2" "tile-0-3" "tile-0-4" "tile-0-5" "tile-0-6" "tile-0-7")
NCAT_PORT=23457
NCAT_BIN=$HOME/nmap/bin/ncat

cd "${IVS_TEMP_DIR}"
mkdir -p build
cd build

echo " --- Running CMake"
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}

echo " --- Building"
make

echo " --- Booting ncat"
ORIGINAL_HOST=$(cut -d' ' -f1 <<< "${SSH_CLIENT}")
${NCAT_BIN} --exec "${NCAT_BIN} ${ORIGINAL_HOST} ${HOST_PORT}" -l ${NCAT_PORT} --keep-open &
NCAT_PID=$!

echo " --- Connecting to clients..."
SLAVE_PIDS=()
for SLAVE in ${SLAVE_IPS[*]}
do
  echo " --- Connecting to ${SLAVE}"
  ssh -t -t ${SLAVE} "cd ${IVS_TEMP_DIR}/build; ./ivs_tile.sh \"${CLAM3_KERNEL}\" \"${IVS_TEMP_DIR}\" \"${NCAT_PORT}\"" &
  SLAVE_PIDS+=("$!")
  sleep 1
done

echo " --- Waiting for slaves"
wait ${SLAVE_PIDS[*]}

echo " --- Slaves done, killing ncat"
kill ${NCAT_PID}

echo " --- IVS script done"