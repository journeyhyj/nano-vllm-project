#!/usr/bin/env bash
# Runs the bench_prefix_cache.py matrix on one GPU, one engine process at a time (the engine binds a fixed port).
# Usage: benchmarks/run_prefix_cache_matrix.sh <out dir> <policy>... ("-" = no --policy flag, i.e. the engine default;
#        the baseline before hybrid prefix caching existed was run with "-")
# Env: MODEL (default ~/huggingface/Qwen3.5-4B), GPU (default 0 = RTX 4090 at PCI 1A:00.0 with CUDA_DEVICE_ORDER=PCI_BUS_ID),
#      REPEATS (default 2), R0 (number of the first repeat, default 1), ONLY (optional regex over run names),
#      EXTRA (extra bench flags), PYTHONPATH (which engine checkout to import; recorded in each log header)
set -euo pipefail
OUT=$1; shift
MODEL=${MODEL:-~/huggingface/Qwen3.5-4B}
REPEATS=${REPEATS:-2}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

# name | bench flags. Shared system prompt: 64 requests, 128 new tokens, two arrival rates per prompt length
# (about 20% and 55% of the no-cache prefill capacity of a 4090). Multi-turn: 16 conversations x 6 turns, 256 new tokens.
RUNS=(
  "shared2000_qps1.5|--workload shared --prefix 2000 --qps 1.5"
  "shared2000_qps4|--workload shared --prefix 2000 --qps 4"
  "shared2048_qps4|--workload shared --prefix 2048 --qps 4"
  "shared4000_qps0.75|--workload shared --prefix 4000 --qps 0.75"
  "shared4000_qps2|--workload shared --prefix 4000 --qps 2"
  "shared8000_qps0.4|--workload shared --prefix 8000 --qps 0.4"
  "shared8000_qps1|--workload shared --prefix 8000 --qps 1"
  "multiturn2000_qps0.5|--workload multiturn --prefix 2000 --qps 0.5 --num-convs 16 --turns 6 --max-tokens 256"
)

for policy in "$@"; do
  for run in "${RUNS[@]}"; do
    name=${run%%|*}; flags=${run#*|}
    if [[ -n "${ONLY:-}" && ! "$name" =~ $ONLY ]]; then continue; fi
    pflag=""; ptag="default"
    if [[ "$policy" != "-" ]]; then pflag="--policy $policy"; ptag=$policy; fi
    for r in $(seq "${R0:-1}" $(( ${R0:-1} + REPEATS - 1 ))); do
      tag="${name}_${ptag}_r${r}"
      cmd="python benchmarks/bench_prefix_cache.py $MODEL $flags $pflag ${EXTRA:-} --out $OUT/$tag.json"
      {
        echo "# $(date -Is) host=$(hostname) git=$(git rev-parse --short HEAD) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
        echo "# CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH=${PYTHONPATH:-}"
        nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/# gpu: /' || true
        echo "# $cmd"
      } > "$OUT/$tag.log"
      echo "[$(date +%T)] $tag"
      eval "$cmd" >> "$OUT/$tag.log" 2>&1 || echo "FAILED: $tag (see $OUT/$tag.log)"
    done
  done
done
