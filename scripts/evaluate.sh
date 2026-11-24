#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${EVALUATION_DIR:?set EVALUATION_DIR to an output directory}"

cd "$HERE/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export REWARD_CKPTS="${REWARD_CKPTS:-${ASSET_ROOT:-$PWD/assets}/reward_ckpts}"
export OCR_ROOT="${OCR_ROOT:-${ASSET_ROOT:-$PWD/assets}/ocr}"
export GENEVAL_CONFIG="${GENEVAL_CONFIG:-${ASSET_ROOT:-$PWD/assets}/geneval/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py}"
export GENEVAL_CLIP_CACHE="${GENEVAL_CLIP_CACHE:-${ASSET_ROOT:-$PWD/assets}/geneval}"
cfg="${EVAL_CFG:-4.5}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"

options=()
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
  options+=(--checkpoint_path "$CHECKPOINT_DIR")
fi
if [[ "$SAVE_IMAGES" == 1 ]]; then
  options+=(--save_images)
fi

for dataset in geneval ocr drawbench; do
  python="${PYTHON:-python}"
  case "$dataset" in
    geneval) python="${GENEVAL_EVAL_PYTHON:-$python}" ;;
    ocr) python="${OCR_EVAL_PYTHON:-$python}" ;;
  esac
  "$python" -m torch.distributed.run --standalone \
    --nproc_per_node "$NUM_PROCESSES" \
    scripts/evaluation.py \
    --model_type sd3 \
    "${options[@]}" \
    --dataset "$dataset" \
    --output_dir "$EVALUATION_DIR/$dataset" \
    --mixed_precision fp16 \
    --resolution 512 --num_inference_steps 40 --guidance_scale "$cfg"
done
