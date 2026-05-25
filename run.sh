#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-dog_bike_man.jpg}
MODEL=${MODEL:-nano}
BACKEND=${BACKEND:-wgpu}
WARMUP=${WARMUP:-5}
RUNS=${RUNS:-20}
OUTPUT=${OUTPUT:-output.jpg}
CSV=${CSV:-burn_${BACKEND}.csv}

cargo run --release --example infer -- \
  "$IMAGE" \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --output "$OUTPUT" \
  --warmup "$WARMUP" \
  --test-runs "$RUNS" \
  --test-csv "$CSV"
