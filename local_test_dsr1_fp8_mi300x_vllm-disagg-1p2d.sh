#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# 1P+2D Disaggregated vLLM — SSH + Docker launcher (no Slurm)
#
# Launches server.sh inside Docker on three nodes via SSH:
#   Node 0 (PREFILL_NODE):    proxy + prefill (kv_producer) + benchmark
#   Node 1 (DECODE_NODE_1):   decode (kv_consumer)
#   Node 2 (DECODE_NODE_2):   decode (kv_consumer)
#
# Prerequisites:
#   - Passwordless SSH from this machine to all three nodes
#   - Docker image pulled on all nodes
#   - Model weights at the configured host dirs on each node
#   - InferenceX repo at INFERENCEX_DIR on all nodes
#
# Usage:
#   bash local_test_dsr1_fp8_mi355x_vllm-disagg-1p2d.sh
#   ISL=8192 OSL=1024 CONC_LIST="8 16 32" bash local_test_dsr1_fp8_mi355x_vllm-disagg-1p2d.sh
# ──────────────────────────────────────────────────────────────

# --- Configurable parameters (override via env) ---
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:v0.19.0}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-0528}"
PRECISION="${PRECISION:-fp8}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-64 128 256}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

VLLM_MORIIO_CONNECTOR_READ_MODE="${VLLM_MORIIO_CONNECTOR_READ_MODE:-1}"
PROXY_STREAM_IDLE_TIMEOUT="${PROXY_STREAM_IDLE_TIMEOUT:-300}"

# 1P + 2D topology: each node uses all 8 GPUs with TP=8
PREFILL_NODE="${PREFILL_NODE:-45.63.78.168}"
DECODE_NODE_1="${DECODE_NODE_1:-107.191.51.58}"
DECODE_NODE_2="${DECODE_NODE_2:-144.202.60.213}"
PREFILL_TP="${PREFILL_TP:-8}"
DECODE_TP="${DECODE_TP:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Per-node host paths containing the model directory.
# Mounted as /models inside Docker; model accessed at /models/<MODEL_DIR_NAME>.
PREFILL_MODEL_HOST_DIR="${PREFILL_MODEL_HOST_DIR:-/mnt}"
DECODE1_MODEL_HOST_DIR="${DECODE1_MODEL_HOST_DIR:-/mnt}"
DECODE2_MODEL_HOST_DIR="${DECODE2_MODEL_HOST_DIR:-/mnt}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-DeepSeek-R1-0528}"

INFERENCEX_DIR="${INFERENCEX_DIR:-${HOME}/InferenceX}"

SSH_USER="${SSH_USER:-$(whoami)}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ${HOME}/.ssh/id_ed25519 -o IdentitiesOnly=yes"
DRY_RUN="${DRY_RUN:-0}"

# --- Derived ---
IPADDRS="${PREFILL_NODE},${DECODE_NODE_1},${DECODE_NODE_2}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="local_${TIMESTAMP}"
CONC_X="${CONC_LIST// /x}"
DOCKER_MODEL_PATH="/models/${MODEL_DIR_NAME}"

LOG_DIR="${HOME}/logs/vllm_disagg"
mkdir -p "$LOG_DIR"
BENCHMARK_LOGS_DIR="${LOG_DIR}/benchmark_logs_${TIMESTAMP}"
mkdir -p "$BENCHMARK_LOGS_DIR"

CONT_NAME="vllm_disagg_${SSH_USER}_${JOB_ID}"

echo "============================================"
echo " DeepSeek-R1 FP8 vLLM Disagg 1P+2D"
echo "============================================"
echo " Image:          ${IMAGE}"
echo " Model:          ${MODEL_NAME}"
echo " ISL/OSL:        ${ISL}/${OSL}"
echo " Concurrency:    ${CONC_LIST}"
echo " Prefill node:   ${PREFILL_NODE} (TP=${PREFILL_TP})"
echo "   model:        ${PREFILL_MODEL_HOST_DIR}/${MODEL_DIR_NAME}"
echo " Decode node 1:  ${DECODE_NODE_1} (TP=${DECODE_TP})"
echo "   model:        ${DECODE1_MODEL_HOST_DIR}/${MODEL_DIR_NAME}"
echo " Decode node 2:  ${DECODE_NODE_2} (TP=${DECODE_TP})"
echo "   model:        ${DECODE2_MODEL_HOST_DIR}/${MODEL_DIR_NAME}"
echo " Docker model:   ${DOCKER_MODEL_PATH}"
echo " Repo:           ${INFERENCEX_DIR}"
echo " Logs:           ${BENCHMARK_LOGS_DIR}"
echo " Container:      ${CONT_NAME}"
echo "============================================"

# ──────────────────────────────────────────────────────────────
# Build docker run command for a given node rank
# ──────────────────────────────────────────────────────────────
build_docker_cmd() {
    local node_rank=$1
    local model_host_dir=$2

    cat <<EOF
set -e
docker rm -f ${CONT_NAME} 2>/dev/null || true
exec docker run --rm \\
    --name ${CONT_NAME} \\
    --init \\
    --stop-timeout 10 \\
    --device /dev/dri \\
    --device /dev/kfd \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    --network host \\
    --ipc host \\
    --group-add video \\
    --cap-add SYS_PTRACE \\
    --security-opt seccomp=unconfined \\
    --privileged \\
    -v /sys:/sys \\
    -v ${model_host_dir}:/models \\
    -v ${INFERENCEX_DIR}:/workspace \\
    --shm-size 128G \\
    -v /tmp:/run_logs \\
    -v ${BENCHMARK_LOGS_DIR}:/benchmark_logs \\
    -e SLURM_JOB_ID=${JOB_ID} \\
    -e SLURM_JOB_NODELIST=${IPADDRS} \\
    -e NNODES=3 \\
    -e NODE_RANK=${node_rank} \\
    -e NODE0_ADDR=${PREFILL_NODE} \\
    -e MODEL_DIR=/models \\
    -e MODEL_NAME=${MODEL_NAME} \\
    -e MODEL_PATH=${DOCKER_MODEL_PATH} \\
    -e VLLM_WS_PATH=/workspace/benchmarks/multi_node/vllm_disagg_utils \\
    -e GPUS_PER_NODE=${GPUS_PER_NODE} \\
    -e xP=1 \\
    -e yD=2 \\
    -e IPADDRS=${IPADDRS} \\
    -e BENCH_INPUT_LEN=${ISL} \\
    -e BENCH_OUTPUT_LEN=${OSL} \\
    -e BENCH_RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO} \\
    -e BENCH_NUM_PROMPTS_MULTIPLIER=10 \\
    -e BENCH_MAX_CONCURRENCY=${CONC_X} \\
    -e BENCH_REQUEST_RATE=inf \\
    -e TQDM_MININTERVAL=20 \\
    -e DRY_RUN=${DRY_RUN} \\
    -e BENCHMARK_LOGS_DIR=/benchmark_logs \\
    -e UCX_TLS=tcp,self,shm,rocm_ipc,rocm_copy,cma \\
    -e UCX_SOCKADDR_TLS_PRIORITY=tcp \\
    -e UCX_MEMTYPE_CACHE=y \\
    -e UCX_RNDV_SCHEME=get_zcopy \\
    -e UCX_RNDV_THRESH=4k \\
    -e UCX_ROCM_IPC_MIN_ZCOPY=0 \\
    -e UCX_LOG_LEVEL=warn \\
    -e HSA_ENABLE_SDMA=1 \\
    -e PROXY_STREAM_IDLE_TIMEOUT=${PROXY_STREAM_IDLE_TIMEOUT} \\
    -e VLLM_MORIIO_CONNECTOR_READ_MODE=${VLLM_MORIIO_CONNECTOR_READ_MODE} \\
    -e PYTHONPYCACHEPREFIX=/tmp/pycache \\
    -e PREFILL_ENABLE_EP=false \\
    -e PREFILL_ENABLE_DP=false \\
    -e DECODE_ENABLE_EP=false \\
    -e DECODE_ENABLE_DP=false \\
    -e PREFILL_TP=${PREFILL_TP} \\
    -e DECODE_TP=${DECODE_TP} \\
    --entrypoint "" \\
    ${IMAGE} \\
    bash -lc "mkdir -p /run_logs/slurm_job-${JOB_ID} && /workspace/benchmarks/multi_node/vllm_disagg_utils/server.sh 2>&1 | tee /run_logs/slurm_job-${JOB_ID}/server_rank${node_rank}.log"
EOF
}

# ──────────────────────────────────────────────────────────────
# Cleanup: kill Docker containers on all nodes on exit/interrupt
# ──────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Cleaning up containers on all three nodes..."
    ssh ${SSH_OPTS} "${SSH_USER}@${DECODE_NODE_1}" "docker rm -f ${CONT_NAME} 2>/dev/null || true" 2>/dev/null &
    ssh ${SSH_OPTS} "${SSH_USER}@${DECODE_NODE_2}" "docker rm -f ${CONT_NAME} 2>/dev/null || true" 2>/dev/null &
    ssh ${SSH_OPTS} "${SSH_USER}@${PREFILL_NODE}" "docker rm -f ${CONT_NAME} 2>/dev/null || true" 2>/dev/null &
    wait 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT INT TERM

# ──────────────────────────────────────────────────────────────
# Launch Docker on all three nodes via SSH
# ──────────────────────────────────────────────────────────────
echo ""
echo "Launching decode-1 on ${DECODE_NODE_1} (NODE_RANK=1)..."
DECODE1_CMD=$(build_docker_cmd 1 "$DECODE1_MODEL_HOST_DIR")
ssh ${SSH_OPTS} "${SSH_USER}@${DECODE_NODE_1}" bash -s <<<"$DECODE1_CMD" \
    > "${BENCHMARK_LOGS_DIR}/ssh_decode1_${DECODE_NODE_1}.log" 2>&1 &
DECODE1_SSH_PID=$!

echo "Launching decode-2 on ${DECODE_NODE_2} (NODE_RANK=2)..."
DECODE2_CMD=$(build_docker_cmd 2 "$DECODE2_MODEL_HOST_DIR")
ssh ${SSH_OPTS} "${SSH_USER}@${DECODE_NODE_2}" bash -s <<<"$DECODE2_CMD" \
    > "${BENCHMARK_LOGS_DIR}/ssh_decode2_${DECODE_NODE_2}.log" 2>&1 &
DECODE2_SSH_PID=$!

sleep 2

echo "Launching prefill + proxy on ${PREFILL_NODE} (NODE_RANK=0)..."
PREFILL_CMD=$(build_docker_cmd 0 "$PREFILL_MODEL_HOST_DIR")
ssh ${SSH_OPTS} "${SSH_USER}@${PREFILL_NODE}" bash -s <<<"$PREFILL_CMD" \
    2>&1 | tee "${BENCHMARK_LOGS_DIR}/ssh_prefill_${PREFILL_NODE}.log" &
PREFILL_SSH_PID=$!

echo ""
echo "All three nodes launched. Waiting for prefill node to complete benchmark..."
echo "  Prefill log:  ${BENCHMARK_LOGS_DIR}/ssh_prefill_${PREFILL_NODE}.log"
echo "  Decode1 log:  ${BENCHMARK_LOGS_DIR}/ssh_decode1_${DECODE_NODE_1}.log"
echo "  Decode2 log:  ${BENCHMARK_LOGS_DIR}/ssh_decode2_${DECODE_NODE_2}.log"
echo "──────────────────────────────────────────────"
echo ""

# Wait for prefill node (runs benchmark, then kills proxy → decode exits)
wait $PREFILL_SSH_PID 2>/dev/null
PREFILL_RC=$?

# Give decode nodes a moment to detect proxy shutdown and exit cleanly
sleep 5
wait $DECODE1_SSH_PID 2>/dev/null || true
wait $DECODE2_SSH_PID 2>/dev/null || true

echo ""
echo "──────────────────────────────────────────────"
echo " Benchmark complete (prefill exit code: ${PREFILL_RC})"
echo " Logs: ${BENCHMARK_LOGS_DIR}/"
echo "──────────────────────────────────────────────"
