#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-dog_bike_man.jpg}
MODEL=${MODEL:-nano}
BACKEND=${BACKEND:-flex}
RUNS=${RUNS:-20}
OUTPUT=${OUTPUT:-output.jpg}
CSV=${CSV:-burn_flex.csv}

cargo run --release --example infer -- \
  "$IMAGE" \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --output "$OUTPUT" \
  --test-runs "$RUNS" \
  --test-csv "$CSV"
