#!/usr/bin/env bash
set -euo pipefail

# ── ADE20K Semantic ──────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1
CKPT_IN="old_ckpts/ade_sem_vit_l_next_mf_512_last.ckpt"
CKPT_OUT="ckpts/ade20k_semantic_pmt_large_512.ckpt"
python3 scripts/convert_ckpt.py "$CKPT_IN" "$CKPT_OUT"
python3 main.py validate --config configs/ade20k/semantic/pmt.yaml \
    --trainer.devices 1 \
    --model.init_args.ckpt_path "$CKPT_OUT" \
    --model.network.masked_attn_enabled False \
    --data.path /home/ncavagnero/datasets/ade20k

# ── COCO Panoptic ViT-L 640 ─────────────────────────────────────
# 
#export CUDA_VISIBLE_DEVICES=1
#CKPT_IN="old_ckpts/coco_pan_vit_l_next_mf_640_last_3.ckpt"
#CKPT_OUT="ckpts/coco_panoptic_pmt_large_640.ckpt"
#python3 scripts/convert_ckpt.py "$CKPT_IN" "$CKPT_OUT"
#python3 main.py validate --config configs/coco/panoptic/pmt_640.yaml \
#    --trainer.devices 1 \
#    --model.init_args.ckpt_path "$CKPT_OUT" \
#    --data.path /home/ncavagnero/datasets/coco

# ── COCO Panoptic ViT-S 640 ─────────────────────────────────────
#export CUDA_VISIBLE_DEVICES=0
#CKPT_IN="old_ckpts/coco_pan_vit_s_next_mf_640_last.ckpt"
#CKPT_OUT="ckpts/coco_panoptic_pmt_small_640.ckpt"
#python3 scripts/convert_ckpt.py "$CKPT_IN" "$CKPT_OUT"
#python3 main.py validate --config configs/coco/panoptic/pmt_vits.yaml \
#    --trainer.devices 1 \
#    --model.init_args.ckpt_path "$CKPT_OUT" \
#    --data.path /home/ncavagnero/datasets/coco

# ── COCO Panoptic ViT-B 640 ─────────────────────────────────────
#export CUDA_VISIBLE_DEVICES=3
#CKPT_IN="old_ckpts/coco_pan_vit_b_next_mf_640_last.ckpt"
#CKPT_OUT="ckpts/coco_panoptic_pmt_base_640.ckpt"
#python3 scripts/convert_ckpt.py "$CKPT_IN" "$CKPT_OUT"
#python3 main.py validate --config configs/coco/panoptic/pmt_vitb.yaml \
#    --trainer.devices 1 \
#    --model.init_args.ckpt_path "$CKPT_OUT" \
#    --data.path /home/ncavagnero/datasets/coco

# ── COCO Panoptic ViT-L 1280 ────────────────────────────────────
#export CUDA_VISIBLE_DEVICES=2
#CKPT_IN="old_ckpts/coco_pan_vit_l_next_mf_1280_last.ckpt"
#CKPT_OUT="ckpts/coco_panoptic_pmt_large_1280.ckpt"
#python3 scripts/convert_ckpt.py "$CKPT_IN" "$CKPT_OUT"
#python3 main.py validate --config configs/coco/panoptic/pmt_1280.yaml \
#    --trainer.devices 1 \
#    --model.init_args.ckpt_path "$CKPT_OUT" \
#    --data.path /home/ncavagnero/datasets/coco \
#    --data.batch_size 2

# ── COCO Instance ViT-L 640 ─────────────────────────────────────
#export CUDA_VISIBLE_DEVICES=0
#CKPT_IN="old_ckpts/coco_ins_vit_l_next_mf_640_last.ckpt"
#CKPT_OUT="ckpts/coco_instance_pmt_large_640.ckpt"
#python3 scripts/convert_ckpt.py "$CKPT_IN" "$CKPT_OUT"
#python3 main.py validate --config configs/coco/instance/pmt_640.yaml \
#    --trainer.devices 1 \
#    --model.init_args.ckpt_path "$CKPT_OUT" \
#    --data.path /home/ncavagnero/datasets/coco

# ── COCO Instance ViT-L 1280 ────────────────────────────────────
#export CUDA_VISIBLE_DEVICES=3
#CKPT_IN="old_ckpts/coco_ins_vit_l_next_mf_1280_last.ckpt"
#CKPT_OUT="ckpts/coco_instance_pmt_large_1280.ckpt"
#python3 scripts/convert_ckpt.py "$CKPT_IN" "$CKPT_OUT"
#python3 main.py validate --config configs/coco/instance/pmt_1280.yaml \
#    --trainer.devices 1 \
#    --model.init_args.ckpt_path "$CKPT_OUT" \
#    --data.path /home/ncavagnero/datasets/coco \
#    --data.batch_size 2
