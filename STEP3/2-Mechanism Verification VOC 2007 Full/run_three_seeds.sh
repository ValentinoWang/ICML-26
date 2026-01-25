#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/ICML/2-Mechanism Verification VOC 2007 Full/Results/YOLOv8n_baseline/voc07_noisy"
MODEL="/root/yolov8n.pt"
DATA="/root/autodl-tmp/dataset/voc07_noisy/voc07_noisy.yaml"
COMMON_ARGS="model=$MODEL data=$DATA epochs=10000 batch=128 imgsz=640 optimizer=AdamW lr0=0.001 patience=20 workers=4 device=0 deterministic=True save_period=-1 verbose=True plots=True"
SEEDS=(1088)

for s in "${SEEDS[@]}"; do
  echo ">>> Training YOLOv8n on voc07_noisy with seed $s"
  yolo detect train $COMMON_ARGS project="$ROOT/seed_$s" name=results seed="$s"
done
