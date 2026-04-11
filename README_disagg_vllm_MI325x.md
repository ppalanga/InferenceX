# Disaggregated vLLM Inference — DeepSeek-R1-0528 FP8 on MI325X

Run disaggregated vLLM with separate prefill and decode nodes using MoRI-IO
on AMD MI325X GPUs (8 GPUs per node, TP=8).

Two topologies are provided:

| Script | Topology | Nodes |
|--------|----------|-------|
| `local_test_dsr1_fp8_mi355x_vllm-disagg.sh` | 1 Prefill + 1 Decode | 2 |
| `local_test_dsr1_fp8_mi355x_vllm-disagg-1p2d.sh` | 1 Prefill + 2 Decode | 3 |

## Prerequisites

- 2 or 3 bare-metal nodes, each with 8x AMD MI325X GPUs
- Docker installed on all nodes
- RDMA-capable NICs (Mellanox ConnectX) for KV transfer
- Same Linux user account on all nodes

## Step 1: Set Up Passwordless SSH

All scripts launch Docker containers on remote nodes via SSH. You need
passwordless SSH from the **driver node** (where you run the script) to
**every node**, including itself.

### Generate an SSH key (on the driver node)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
```

### Copy the public key to every node

```bash
# Show the key
cat ~/.ssh/id_ed25519.pub

# On EACH node (including the driver), append to authorized_keys:
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "<paste-public-key-here>" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### Verify connectivity

```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes user@<NODE_IP> hostname
```

## Step 2: Open Firewall Between Nodes

The services use many ports (etcd 2379, barrier 5000, vLLM 2584, proxy
30000/36367, RDMA/UCX dynamic ports). The simplest approach is to allow all
traffic between peer nodes by source IP using `ufw`:

```bash
# On EACH node, allow traffic from every OTHER node:
sudo ufw allow from <PEER_IP_1>
sudo ufw allow from <PEER_IP_2>
# ... repeat for all peers
```

For example, with three nodes A, B, C:
- On A: `sudo ufw allow from <B>` and `sudo ufw allow from <C>`
- On B: `sudo ufw allow from <A>` and `sudo ufw allow from <C>`
- On C: `sudo ufw allow from <A>` and `sudo ufw allow from <B>`

If `ufw` is not active (`sudo ufw status` shows "inactive"), skip this step.

## Step 3: Pull the Docker Image (All Nodes)

```bash
docker pull vllm/vllm-openai-rocm:v0.19.0
```

Run this on **every node**. For remote nodes:

```bash
ssh user@<NODE_IP> "docker pull vllm/vllm-openai-rocm:v0.19.0"
```

## Step 4: Clone the InferenceX Repository (All Nodes)

The repository must be present at the same path on all nodes (default
`~/InferenceX`). Make sure to check out the `amd/vllm_disagg_mvp_dev` branch.

```bash
# On the driver node
git clone -b amd/vllm_disagg_mvp_dev https://github.com/SemiAnalysisAI/InferenceX.git ~/InferenceX

# Copy to remote nodes
scp -r ~/InferenceX user@<NODE_IP>:~/InferenceX
```

## Step 5: Download the Model Weights

Download **DeepSeek-R1-0528** (FP8, ~642 GB, 163 safetensors shards) to each
node. The host directory can differ per node — the scripts mount it as
`/models` inside Docker.

Recommended locations:
- `/dev/shm/Deepseek-R1-0528` — fastest (tmpfs, requires sufficient RAM)
- `/mnt/Deepseek-R1-0528` — persistent storage

```bash
# Option A: Download with huggingface-cli
pip install -U huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-R1-0528 \
    --local-dir /dev/shm/Deepseek-R1-0528

# Option B: Copy from another node
rsync -avh --progress user@<SOURCE_IP>:/dev/shm/Deepseek-R1-0528/ \
    /dev/shm/Deepseek-R1-0528/
```

Verify the model is complete (should show 163 shards):

```bash
ls /dev/shm/Deepseek-R1-0528/*.safetensors | wc -l
```

## Step 6: Configure and Run

### 1P+1D (2 nodes)

Edit the node IPs and model paths at the top of
`local_test_dsr1_fp8_mi355x_vllm-disagg.sh`, or override via environment
variables:

```bash
PREFILL_NODE=<PREFILL_IP> \
DECODE_NODE=<DECODE_IP> \
PREFILL_MODEL_HOST_DIR=/dev/shm \
DECODE_MODEL_HOST_DIR=/dev/shm \
ISL=1024 OSL=1024 CONC_LIST="64" \
bash local_test_dsr1_fp8_mi355x_vllm-disagg.sh
```

### 1P+2D (3 nodes)

```bash
PREFILL_NODE=<PREFILL_IP> \
DECODE_NODE_1=<DECODE1_IP> \
DECODE_NODE_2=<DECODE2_IP> \
PREFILL_MODEL_HOST_DIR=/dev/shm \
DECODE1_MODEL_HOST_DIR=/dev/shm \
DECODE2_MODEL_HOST_DIR=/dev/shm \
ISL=1024 OSL=1024 CONC_LIST="64" \
bash local_test_dsr1_fp8_mi355x_vllm-disagg-1p2d.sh
```

### Concurrency Sweep

Pass multiple values to `CONC_LIST` to benchmark at several concurrency
levels in a single run:

```bash
ISL=1024 OSL=1024 CONC_LIST="32 64 128 256" bash local_test_dsr1_fp8_mi355x_vllm-disagg.sh
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE` | `vllm/vllm-openai-rocm:v0.19.0` | Docker image |
| `MODEL_NAME` | `DeepSeek-R1-0528` | Model identifier for config lookup |
| `ISL` | `1024` | Input sequence length |
| `OSL` | `1024` | Output sequence length |
| `CONC_LIST` | `64 128 256` | Space-separated concurrency levels |
| `RANDOM_RANGE_RATIO` | `0.8` | Min ratio for random length sampling |
| `PREFILL_NODE` | — | IP address of the prefill node |
| `DECODE_NODE` | — | IP address of the decode node (1P1D) |
| `DECODE_NODE_1` / `DECODE_NODE_2` | — | Decode node IPs (1P2D) |
| `PREFILL_MODEL_HOST_DIR` | `/mnt` | Host path to model dir on prefill |
| `DECODE_MODEL_HOST_DIR` | `/dev/shm` | Host path to model dir on decode (1P1D) |
| `DECODE1_MODEL_HOST_DIR` / `DECODE2_MODEL_HOST_DIR` | `/dev/shm` | Host paths (1P2D) |
| `MODEL_DIR_NAME` | `Deepseek-R1-0528` | Directory name under the host path |
| `INFERENCEX_DIR` | `~/InferenceX` | Path to InferenceX repo on all nodes |
| `PREFILL_TP` / `DECODE_TP` | `8` | Tensor parallelism per node |
| `GPUS_PER_NODE` | `8` | Number of GPUs per node |
| `SSH_USER` | `$(whoami)` | SSH username for remote nodes |
| `DRY_RUN` | `0` | Set to `1` to skip the actual benchmark |

## Monitoring a Running Benchmark

The scripts stream the prefill node's output to stdout. For long runs, use
`tmux` or `screen`:

```bash
tmux new -s bench
ISL=1024 OSL=1024 bash local_test_dsr1_fp8_mi355x_vllm-disagg.sh
# Ctrl-b d to detach, tmux attach -t bench to reattach
```

Decode node logs are written to the benchmark logs directory:

```bash
# Find the latest log directory
ls -td ~/logs/vllm_disagg/benchmark_logs_* | head -1

# Tail the decode log
tail -f ~/logs/vllm_disagg/benchmark_logs_<TIMESTAMP>/ssh_decode_<IP>.log
```

## Expected Timeline

| Phase | Duration |
|-------|----------|
| Container startup + dependency setup | ~2 min (first run ~5 min) |
| Container creation barrier + grace | ~5 min |
| etcd + proxy startup | ~30 sec |
| Model loading (163 shards) | ~40 sec |
| CUDA graph capture + kernel compile | ~1 min |
| Warmup (128 requests) | ~1 min |
| Benchmark (per concurrency level) | ~5 min |

Total wall time: ~15 min for a single concurrency level, ~25-30 min for a
3-level sweep.

## Troubleshooting

### SSH: "Permission denied"

- Verify `~/.ssh/authorized_keys` contains the public key on the target node
- Check permissions: `chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys`
- Test with verbose mode: `ssh -vvv -i ~/.ssh/id_ed25519 user@<IP>`

### Barrier timeout: "Waiting for nodes... ERROR: Timeout"

- Check the firewall: `sudo ufw status` — ensure peer IPs are allowed
- Test port connectivity: `python3 -c "import socket; s=socket.create_connection(('<IP>', 5000), 5); print('OK'); s.close()"`
- Check the decode log for errors: `tail -50 ~/logs/vllm_disagg/benchmark_logs_*/ssh_decode_*.log`

### etcd unhealthy

- Usually caused by firewall blocking port 2379 between nodes
- Run `sudo ufw allow from <PEER_IP>` on the node running etcd (prefill node)

### Container killed / "Canceled: grpc"

- A previous run's cleanup may have collided. Kill stale containers on all nodes:
  ```bash
  ssh user@<IP> "docker rm -f \$(docker ps -aq) 2>/dev/null"
  ```
- Then relaunch the script

### Slow first run

The first run installs dependencies (UCX, RIXL, etcd, MoRI) inside the
container. Subsequent runs reuse the cached installations and start much
faster.
