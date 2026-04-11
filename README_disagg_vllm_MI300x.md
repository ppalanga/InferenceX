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

## Topology

```
┌──────────────────┐
│  Prefill (Node 0)│──── proxy + kv_producer + benchmark
└──────┬───────────┘
       │  KV transfer (RDMA)
┌──────┴───────────┐
│  Decode 1 (Node 1)│──── kv_consumer
└──────────────────┘
┌──────────────────┐
│  Decode 2 (Node 2)│──── kv_consumer
└──────────────────┘
```

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

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Permission denied` on SSH | Public key not installed on target node | Re-run `ssh-copy-id`; check `~/.ssh/authorized_keys` permissions (600) and `~/.ssh` (700) |
| Barrier stuck at "Waiting for nodes" on port 5000 | Firewall blocking inter-node traffic | Run `iptables -I INPUT -s <IP> -j ACCEPT` on all nodes for each peer IP |
| etcd "unhealthy cluster" / barrier stuck on port 2379 | Firewall blocking etcd ports | Same firewall fix as above — whitelist all peer IPs |
| Docker image pull slow on first run | Large image (~30 GB) | Pre-pull with `docker pull` on all nodes before running |
| Model directory not found | Wrong `MODEL_DIR_NAME` or host path | Check that `/mnt/DeepSeek-R1-0528` (or your configured path) exists on all nodes |
| `server.sh` not found inside container | InferenceX repo missing on a node | `rsync` the repo to all nodes (Step 5) |
