#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"<prompt>\" [outdir] [seed] [steps] [samples] [scale]"
  echo "Example: $0 \"a portrait photo of a cat astronaut\" outputs/original_prompt 42 50 1 7.5"
  exit 1
fi

PROMPT="$1"
OUTDIR="${2:-outputs/original_prompt}"
SEED="${3:-42}"
STEPS="${4:-50}"
SAMPLES="${5:-1}"
SCALE="${6:-7.5}"

CONFIG_PATH="configs/stable-diffusion/v1-inference.yaml"
CKPT_PATH="models/sd-v1-4-full-ema.ckpt"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config not found at $CONFIG_PATH"
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Error: checkpoint not found at $CKPT_PATH"
  echo "Download SD v1.4 original ckpt and place it at that path."
  exit 1
fi

mkdir -p "$OUTDIR"

python scripts/txt2img.py \
  --prompt "$PROMPT" \
  --outdir "$OUTDIR" \
  --config "$CONFIG_PATH" \
  --ckpt "$CKPT_PATH" \
  --ddim_steps "$STEPS" \
  --n_iter 1 \
  --n_samples "$SAMPLES" \
  --scale "$SCALE" \
  --seed "$SEED" \
  --precision autocast

