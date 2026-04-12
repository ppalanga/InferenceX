# Combined guides: SGLang PD and disaggregated vLLM (MI300X / MI325X)

## Document structure

This file is three guides in sequence: **SGLang** prefill–decode disaggregation (MI300X and MI325X), then **vLLM** disaggregation on **MI325X**, then **vLLM** on **MI300X**. 

1. **SGLang prefill–decode (PD) disaggregation on AMD MI300X / MI325X**   


2. **PD Disaggregated vLLM Inference — DeepSeek-R1-0528 FP8 on MI325X** 
 

3. **DeepSeek-R1 FP8 — 1P+2D Disaggregated vLLM Benchmark**   

---

# SGLang prefill–decode (PD) disaggregation on AMD MI300X / MI325X

This guide applies to **AMD MI300X** and **MI325X** (and similar CDNA3-class ROCm deployments) running **SGLang disaggregated inference** (prefill and decode on separate GPU nodes) via the **InferenceX** **SSH + Docker** launchers—no Slurm required.

The launcher scripts are `run_1p1d_sglang_mi300_mi325x.sh` and `run_1p2d_sglang_mi300_mi325x.sh` (optional wrapper `start_sglang_pd_mi300_mi325x.sh`). You choose the **container image** (`IMAGE`), **GPU count**, and **host drivers** (e.g. Broadcom `bnxt_re` / libbnxt tarball) to match your hardware.

For the broader InferenceX project (benchmarking across NVIDIA/AMD, CI, and the public dashboard), see the [repository README](../README.md).

---

## 1. What you are running

| Topology | Script | Nodes | Roles |
|----------|--------|-------|--------|
| **1 prefill + 1 decode** | `run_1p1d_sglang_mi300_mi325x.sh` | 2 | Rank 0: prefill, router, benchmark client. Rank 1: decode. |
| **1 prefill + 2 decode** | `run_1p2d_sglang_mi300_mi325x.sh` | 3 | Rank 0: prefill + router + benchmark. Ranks 1–2: decode workers. |

**IP order matters.** The cluster addresses passed to the engine are:

- **1P+1D:** `IPADDRS = prefill_ip,decode_ip` (prefill must be first).
- **1P+2D:** `IPADDRS = prefill_ip,decode1_ip,decode2_ip` (decode indices must align with `benchmarks/multi_node/amd_utils/server.sh`).

The launchers resolve data-plane IPs by SSH’ing each host and running `ip route get 1.1.1.1` (unless you set the IPs explicitly—see below).

---

## 2. Prerequisites checklist

Use this before the first production run.

### 2.1 Control machine (launcher)

- **InferenceX** cloned at a path you will use as `REMOTE_REPO` on **every** GPU node (the same path is bind-mounted into the container as `/workspace`).
- **Passwordless SSH** from the launcher to all GPU nodes, with a key the scripts can use (default: `${HOME}/.ssh/id_rsa`; override with `SSH_OPTS` if needed).
- Optional: **local** check that model directories exist, or set `SKIP_LOCAL_MODEL_CHECK=1` to validate only on the cluster.

### 2.2 Every GPU node

- **Docker** installed; the scripts use `sudo docker` by default (`USE_SUDO_FOR_DOCKER=1`). If your user is in the `docker` group, set `USE_SUDO_FOR_DOCKER=0`.
- **Container image** pulled. Default in the scripts is `lmsysorg/sglang:v0.5.9-rocm700-mi30x` (`IMAGE`). Pick a tag that matches **your** GPU generation and ROCm line; if a build targets one SKU explicitly, align with vendor or SGLang release notes.
- **Model weights** on local disk (often `/dev/shm` or `/mnt`): each node needs `HOST_MODEL_DIR/MODEL_NAME` as a directory (e.g. `DeepSeek-R1-0528`). Prefill and decode can use different host parents via `PREFILL_MODEL_HOST_DIR` and `DECODE_MODEL_HOST_DIR` (and `DECODE_MODEL_HOST_DIR_2` for the second decode in 1P+2D).
- **RDMA / InfiniBand device list** for MoRI: set `IBDEVICES` to a comma-separated list of HCAs, or install host tools so `benchmarks/multi_node/amd_utils/detect_ibdevices_bnxt.sh` can populate it automatically.

### 2.3 Network between nodes

Open **TCP** between all participating nodes for the PD and routing stack (defaults used in `server.sh`):

- **5000** — barrier / sync
- **8000** — SGLang PD HTTP endpoints
- **30000** — router / head traffic

Example (host firewall): allow the peer node IPs on these ports (e.g. `ufw` rules from each node to the others).

Do **not** set `DOCKER="sudo docker"` as a single env var for remote runs; use `DOCKER_BIN=docker` and `USE_SUDO_FOR_DOCKER=1` instead—multi-word `DOCKER` breaks SSH `env` passing.

---

## 3. Quick start

### 3.1 Environment file pattern

A minimal pattern (from `start_sglang_pd_mi300_mi325x.sh`) is:

```bash
export PREFILL_MODEL_HOST_DIR="${PREFILL_MODEL_HOST_DIR:-/dev/shm}"
export DECODE_MODEL_HOST_DIR="${DECODE_MODEL_HOST_DIR:-/dev/shm}"
export IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.9-rocm700-mi30x}"

# Optional: rebuild MoRI inside the container (e.g. topology / latest fixes)
export INSTALL_MORI_IN_CONTAINER=1
export INSTALL_MORI_MODE=git
export MORI_GIT_REF=main

# Optional: rebuild libbnxt in-container from Broadcom tarball (match host bnxt_re)
# Use the tarball that matches the bnxt_re version on *this* host (see /usr/src/bnxt_re-* on the node).
# Examples:
#   MI325X-class:  libbnxt_re-231.0.162.0.tar.gz
#   MI300X-class:  libbnxt_re-230.2.52.0.tar.gz
export REBUILD_LIBBNXT_IN_CONTAINER=1
export PATH_TO_BNXT_TAR_PACKAGE=/workspace/driver/libbnxt_re-231.0.162.0.tar.gz
# export PATH_TO_BNXT_TAR_PACKAGE=/workspace/driver/libbnxt_re-230.2.52.0.tar.gz

export PREFILL_NODE="<prefill-host-or-ip>"
export DECODE_NODE="<decode-host>"          # 1P+1D
# or for 1P+2D:
export DECODE_NODE_1="<decode1>"
export DECODE_NODE_2="<decode2>"

cd /path/to/InferenceX
bash run_1p1d_sglang_mi300_mi325x.sh
# or: bash run_1p2d_sglang_mi300_mi325x.sh
```

Driver tarballs can be downloaded from https://www.broadcom.com/support/download-search. Please download the tarball that matches the bnxt_re version on **this** host (see /usr/src/bnxt_re-* on the node). We also provide the tarball under `InferenceX/driver/` on **each** node at the same `REMOTE_REPO` layout so the in-container path `/workspace/driver/...` exists after bind-mount.

### 3.2 Validate without running

```bash
DRY_RUN=1 bash run_1p1d_sglang_mi300_mi325x.sh
```

### 3.3 Common benchmark overrides

```bash
export MODEL_NAME=DeepSeek-R1-0528
export ISL=1024 OSL=1024
export CONC_LIST="8 16"
bash run_1p1d_sglang_mi300_mi325x.sh
```

Parallelism defaults (override as needed): `PREFILL_TP`, `DECODE_TP`, `PREFILL_EP`, `DECODE_EP`, `PREFILL_DP_ATTN`, `DECODE_DP_ATTN`, `DECODE_MTP_SIZE`, `GPUS_PER_NODE`, `xP`, `yD`.

---

## 4. How the launch works (mental model)

1. **Launcher** SSHs to each node with a large `env ... bash /path/to/InferenceX/scripts/_disagg_ssh_remote_inner.sh`.
2. **`_disagg_ssh_remote_inner.sh`** starts one **Docker** container per node: GPUs, `/dev/infiniband`, host network, shared memory, and volumes:
   - `HOST_MODEL_DIR` → `/models`
   - `HOST_REPO` (InferenceX root) → `/workspace`
   - Log directories for benchmarks and `/run_logs`
3. **`_disagg_container_entry.sh`** (inside the image) optionally runs **libbnxt** rebuild, then **MoRI** install, then execs **`benchmarks/multi_node/amd_utils/server.sh`**, which implements the SGLang PD topology and benchmark client for your `MODEL_NAME` (via `models.yaml` in that directory).

Startup order in the 1P+1D script: **decode first** (background), short sleep, then **prefill** (foreground with tee to log). Adjust if you customize scripts.

---

## 5. Configuration reference (environment)

| Variable | Purpose |
|----------|---------|
| `PREFILL_NODE`, `DECODE_NODE` | SSH targets for 1P+1D (`user@host`). |
| `DECODE_NODE_1`, `DECODE_NODE_2` | Second decode host for 1P+2D. |
| `SSH_USER` | Defaults to `whoami` on launcher if not set in script defaults. |
| `REMOTE_REPO` / `INFERENCEX_DIR` | InferenceX root; must match on all nodes for `/workspace`. |
| `PREFILL_MODEL_HOST_DIR`, `DECODE_MODEL_HOST_DIR` | Host paths containing `MODEL_NAME`. |
| `DECODE_MODEL_HOST_DIR_2` | Optional second decode model root (1P+2D). |
| `IMAGE` | SGLang ROCm container tag; **must match your AMD SKU / ROCm** (MI300X vs MI325X, etc.). |
| `MODEL_NAME` | Key in `benchmarks/multi_node/amd_utils/models.yaml`. |
| `IBDEVICES` | RDMA devices for MoRI (required unless auto-detected). |
| `PREFILL_IP`, `DECODE_IP` | Optional explicit data-plane IPs (1P+1D). |
| `PREFILL_IP`, `DECODE1_IP`, `DECODE2_IP` | Optional explicit IPs (1P+2D). |
| `BARRIER_SYNC_PORT`, `SGLANG_PD_PORT`, `ROUTER_PORT` | Override ports if defaults conflict. |
| `MORI_RDMA_TC` | Optional RDMA traffic class. |
| `REBUILD_LIBBNXT_IN_CONTAINER`, `PATH_TO_BNXT_TAR_PACKAGE` | In-container libbnxt build from tarball (**match host NIC driver version**). |
| `INSTALL_MORI_IN_CONTAINER`, `INSTALL_MORI_MODE`, `MORI_GIT_REF`, … | Build MoRI from git or a mounted path. |
| `SKIP_LOCAL_MODEL_CHECK` | `1` to skip launcher-side weight directory check. |
| `USE_SUDO_FOR_DOCKER`, `DOCKER_BIN`, `EXTRA_DOCKER_ARGS`, `DOCKER_SHM_SIZE` | Docker invocation tuning. |

---

## 6. Logs and artifacts

- **Launcher:** `${HOME}/logs/sglang_disagg/benchmark_logs_<timestamp>/`
  - `ssh_prefill_<node>.log`, `ssh_decode_<node>.log` (naming varies for 1P+2D).
- **On each node:** `/tmp/inferencex_disagg_logs_${JOB_ID}` and `/tmp/run_logs_${JOB_ID}` (mapped into the container as documented in the scripts).

Use these logs for MoRI/SGLang startup errors, RDMA (`ibv_devinfo`), and benchmark throughput lines.

---

## 8. Related InferenceX entry points

- **Slurm / CI-style** multi-node benchmarks: `benchmarks/multi_node/dsr1_*_sglang-disagg.sh` and `amd_utils/submit.sh` (different orchestration, same `server.sh` family).
- **Matrix / GitHub Actions**: see `AGENTS.md` and `.github/configs/` for `sglang-disagg` framework definitions.

---


# PD Disaggregated vLLM Inference — DeepSeek-R1-0528 FP8 on MI325X

Run disaggregated vLLM with separate prefill and decode nodes using MoRI-IO
on AMD MI325X GPUs (8 GPUs per node, TP=8).

Two topologies are provided:

| Script | Topology | Nodes |
|--------|----------|-------|
| `local_test_dsr1_fp8_mi325x_vllm-disagg.sh` | 1 Prefill + 1 Decode | 2 |
| `local_test_dsr1_fp8_mi325x_vllm-disagg-1p2d.sh` | 1 Prefill + 2 Decode | 3 |

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
git clone -b amd/vllm_disagg_mvp_dev https://github.com/ppalanga/InferenceX.git ~/InferenceX

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
`local_test_dsr1_fp8_mi325x_vllm-disagg.sh`, or override via environment
variables:

```bash
PREFILL_NODE=<PREFILL_IP> \
DECODE_NODE=<DECODE_IP> \
PREFILL_MODEL_HOST_DIR=/dev/shm \
DECODE_MODEL_HOST_DIR=/dev/shm \
ISL=1024 OSL=1024 CONC_LIST="64" \
bash local_test_dsr1_fp8_mi325x_vllm-disagg.sh
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
bash local_test_dsr1_fp8_mi325x_vllm-disagg-1p2d.sh
```

### Concurrency Sweep

Pass multiple values to `CONC_LIST` to benchmark at several concurrency
levels in a single run:

```bash
ISL=1024 OSL=1024 CONC_LIST="32 64 128 256" bash local_test_dsr1_fp8_mi325x_vllm-disagg.sh
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
ISL=1024 OSL=1024 bash local_test_dsr1_fp8_mi325x_vllm-disagg.sh
# Ctrl-b d to detach, tmux attach -t bench to reattach
```

Decode node logs are written to the benchmark logs directory:

```bash
# Find the latest log directory
ls -td ~/logs/vllm_disagg/benchmark_logs_* | head -1

# Tail the decode log
tail -f ~/logs/vllm_disagg/benchmark_logs_<TIMESTAMP>/ssh_decode_<IP>.log
```

---


# DeepSeek-R1 FP8 — 1P+2D Disaggregated vLLM Benchmark

Run DeepSeek-R1-0528 in a **1-Prefill + 2-Decode** disaggregated configuration
across three MI300X nodes using vLLM with MoRI-IO KV transfer.

## Hardware Requirements

| Role | Count | GPUs per node | Notes |
|------|-------|---------------|-------|
| Prefill | 1 | 8 (TP=8) | Also runs the proxy and benchmark client |
| Decode | 2 | 8 (TP=8) each | KV consumers |

All nodes must have:
- AMD MI300X GPUs (8 per node)
- InfiniBand / RoCE connectivity between nodes
- Docker installed
- Sufficient disk space for the Docker image (~30 GB) and model weights


## Step 1: Assign Node IPs

Identify the IP addresses of your three nodes. In the instructions below we use:

| Variable | Example IP | Role |
|----------|-----------|------|
| `PREFILL_NODE` | 45.63.78.168 | Prefill + proxy |
| `DECODE_NODE_1` | 107.191.51.58 | Decode 1 |
| `DECODE_NODE_2` | 144.202.60.213 | Decode 2 |

Replace these with your actual node IPs throughout.

## Step 2: Set Up Passwordless SSH

The launch script runs from the **prefill node** and SSHes into all three nodes
(including itself) to start Docker containers.

**On the prefill node**, generate an SSH key if one does not already exist:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
```

Copy the public key to **all three nodes** (including the prefill node itself):

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <USER>@<PREFILL_NODE>
ssh-copy-id -i ~/.ssh/id_ed25519.pub <USER>@<DECODE_NODE_1>
ssh-copy-id -i ~/.ssh/id_ed25519.pub <USER>@<DECODE_NODE_2>
```

If `ssh-copy-id` is unavailable, manually append the public key on each node:

```bash
# On each node:
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo '<contents of ~/.ssh/id_ed25519.pub>' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**Verify** passwordless SSH works from the prefill node to all three:

```bash
ssh -i ~/.ssh/id_ed25519 <USER>@<PREFILL_NODE> hostname
ssh -i ~/.ssh/id_ed25519 <USER>@<DECODE_NODE_1> hostname
ssh -i ~/.ssh/id_ed25519 <USER>@<DECODE_NODE_2> hostname
```

## Step 3: Open Firewall Between Nodes

The nodes communicate on multiple ports (etcd: 2379-2380, vLLM: 2584, barriers:
5000, NCCL, RDMA, proxy, etc.). The simplest approach is to whitelist all
traffic between the three nodes.

**On every node**, run:

```bash
sudo iptables -I INPUT -s <PREFILL_NODE> -j ACCEPT
sudo iptables -I INPUT -s <DECODE_NODE_1> -j ACCEPT
sudo iptables -I INPUT -s <DECODE_NODE_2> -j ACCEPT
```

**Verify** connectivity (run from the prefill node):

```bash
# Test a few key ports to each decode node
for PORT in 5000 2379 2380; do
  for IP in <DECODE_NODE_1> <DECODE_NODE_2>; do
    timeout 3 bash -c "echo > /dev/tcp/$IP/$PORT" 2>/dev/null \
      && echo "OK: $IP:$PORT" \
      || echo "BLOCKED: $IP:$PORT (may not be listening yet — OK before launch)"
  done
done
```

> **Note:** Ports may not be listening before the containers start, so
> connection-refused is expected at this stage. What matters is that the
> connections are not blocked by the firewall (i.e., you should not see a
> timeout).

## Step 4: Download Model Weights

The FP8-quantized DeepSeek-R1-0528 model must be present on all three nodes.

**On every node**, download the model to `/mnt`:

```bash
# Using huggingface-cli (install with: pip install huggingface_hub)
huggingface-cli download deepseek-ai/DeepSeek-R1-0528 \
  --local-dir /mnt/DeepSeek-R1-0528
```

The script expects the model directory name to be `DeepSeek-R1-0528` under the
configured host directory. The default mount points are:

| Node | Host path | Docker mount |
|------|-----------|-------------|
| Prefill | `/mnt/DeepSeek-R1-0528` | `/models/DeepSeek-R1-0528` |
| Decode 1 | `/mnt/DeepSeek-R1-0528` | `/models/DeepSeek-R1-0528` |
| Decode 2 | `/mnt/DeepSeek-R1-0528` | `/models/DeepSeek-R1-0528` |

Override per-node paths with `PREFILL_MODEL_HOST_DIR`, `DECODE1_MODEL_HOST_DIR`,
`DECODE2_MODEL_HOST_DIR` if your layout differs.

## Step 5: Clone the InferenceX Repository

The repository must be present at the same path on **all three nodes** (default:
`~/InferenceX`).

**On the prefill node:**

```bash
cd ~
git clone -b amd/vllm_disagg_mvp_dev https://github.com/ppalanga/InferenceX.git InferenceX
```

**Sync to decode nodes** (from the prefill node):

```bash
rsync -az -e "ssh -i ~/.ssh/id_ed25519" \
  ~/InferenceX/ <USER>@<DECODE_NODE_1>:~/InferenceX/

rsync -az -e "ssh -i ~/.ssh/id_ed25519" \
  ~/InferenceX/ <USER>@<DECODE_NODE_2>:~/InferenceX/
```

## Step 6: Pull the Docker Image

Pull the vLLM image on **all three nodes** to avoid slow first-run pulls:

```bash
# On each node (or via SSH from prefill):
docker pull vllm/vllm-openai-rocm:v0.19.0
```

Or from the prefill node via SSH:

```bash
ssh -i ~/.ssh/id_ed25519 <USER>@<PREFILL_NODE> "docker pull vllm/vllm-openai-rocm:v0.19.0" &
ssh -i ~/.ssh/id_ed25519 <USER>@<DECODE_NODE_1> "docker pull vllm/vllm-openai-rocm:v0.19.0" &
ssh -i ~/.ssh/id_ed25519 <USER>@<DECODE_NODE_2> "docker pull vllm/vllm-openai-rocm:v0.19.0" &
wait
```

## Step 7: Configure and Run the Benchmark

Edit the node IPs in the script (or pass them as environment variables):

```bash
cd ~/InferenceX

# Option A: Edit the defaults in the script
vi local_test_dsr1_fp8_mi300x_vllm-disagg-1p2d.sh
# Update PREFILL_NODE, DECODE_NODE_1, DECODE_NODE_2

# Option B: Override via environment variables
export PREFILL_NODE=<your prefill IP>
export DECODE_NODE_1=<your decode1 IP>
export DECODE_NODE_2=<your decode2 IP>
```

Run the benchmark:

```bash
bash local_test_dsr1_fp8_mi300x_vllm-disagg-1p2d.sh
```

### Customizing Benchmark Parameters

Override any parameter via environment variables:

```bash
# Example: different input/output lengths and concurrency levels
ISL=8192 OSL=1024 CONC_LIST="8 16 32" \
  bash local_test_dsr1_fp8_mi300x_vllm-disagg-1p2d.sh
```

Key parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE` | `vllm/vllm-openai-rocm:v0.19.0` | Docker image |
| `PREFILL_NODE` | — | Prefill node IP |
| `DECODE_NODE_1` | — | First decode node IP |
| `DECODE_NODE_2` | — | Second decode node IP |
| `ISL` | `1024` | Input sequence length |
| `OSL` | `1024` | Output sequence length |
| `CONC_LIST` | `64 128 256` | Space-separated concurrency levels |
| `PREFILL_TP` | `8` | Tensor parallelism for prefill |
| `DECODE_TP` | `8` | Tensor parallelism for decode |
| `PREFILL_MODEL_HOST_DIR` | `/mnt` | Host path to model parent dir (prefill) |
| `DECODE1_MODEL_HOST_DIR` | `/mnt` | Host path to model parent dir (decode 1) |
| `DECODE2_MODEL_HOST_DIR` | `/mnt` | Host path to model parent dir (decode 2) |
| `MODEL_DIR_NAME` | `DeepSeek-R1-0528` | Model directory name |
| `SSH_USER` | `$(whoami)` | SSH username for all nodes |
| `DRY_RUN` | `0` | Set to `1` to skip the actual benchmark |

## Monitoring

Logs are written to `~/logs/vllm_disagg/benchmark_logs_<timestamp>/`:

```bash
# Follow prefill output (streamed to terminal by default)
# Decode logs are in separate files:
tail -f ~/logs/vllm_disagg/benchmark_logs_*/ssh_decode1_*.log
tail -f ~/logs/vllm_disagg/benchmark_logs_*/ssh_decode2_*.log
```

You can also check Docker logs directly on any node:

```bash
docker logs -f $(docker ps -q --filter 'name=vllm_disagg')
```

## Cleanup

The script traps `EXIT`, `INT`, and `TERM` to automatically clean up containers
on all nodes. To manually clean up:

```bash
# On each node (or via SSH):
docker rm -f $(docker ps -q --filter 'name=vllm_disagg')
```
