#!/usr/bin/env bash
# Runs bench_snapshot_overhead.py for each policy and prompt length on one GPU, one engine process at a time.
# Usage: benchmarks/run_snapshot_overhead.sh <out dir> <policy>...
# Env: MODEL (default ~/huggingface/Qwen3.5-4B), GPU (default 0 = RTX 4090 at PCI 1A:00.0 with CUDA_DEVICE_ORDER=PCI_BUS_ID),
#      LENS (default "2000 8000"), PYTHONPATH (a worktree, to measure another commit of the engine)
set -euo pipefail
OUT=$1; shift
MODEL=${MODEL:-~/huggingface/Qwen3.5-4B}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
for policy in "$@"; do
  for len in ${LENS:-2000 8000}; do
    tag="len${len}_${policy}"
    cmd="python benchmarks/bench_snapshot_overhead.py $MODEL --policy $policy --len $len --out $OUT/$tag.json"
    {
      echo "# $(date -Is) host=$(hostname) PYTHONPATH=${PYTHONPATH:-} loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
      echo "# CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
      nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/# gpu: /' || true
      echo "# $cmd"
    } > "$OUT/$tag.log"
    echo "[$(date +%T)] $tag"
    eval "$cmd" >> "$OUT/$tag.log" 2>&1 || echo "FAILED: $tag (see $OUT/$tag.log)"
  done
done
