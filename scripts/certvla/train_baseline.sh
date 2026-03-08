#!/bin/bash
# ============================================================================
# CertVLA 基线训练脚本 (OpenVLA-OFT 原版微调)
# ============================================================================
# 用途：在 LIBERO 数据集上微调 OpenVLA-OFT 基线模型（不包含 CertVLA 模块）
# 使用方法：
#   1. 修改下面的 ===== 用户必改参数 ===== 部分
#   2. chmod +x scripts/certvla/train_baseline.sh
#   3. bash scripts/certvla/train_baseline.sh
# ============================================================================

# ===== 用户必改参数 =====
NUM_GPUS=1                                          # GPU 数量
DATA_ROOT="/PATH/TO/modified_libero_rlds"           # RLDS 数据集根目录
RUN_ROOT="/PATH/TO/checkpoints"                     # checkpoint 和日志输出目录
DATASET="libero_spatial_no_noops"                   # 数据集名称，可选：
                                                    #   libero_spatial_no_noops
                                                    #   libero_object_no_noops
                                                    #   libero_goal_no_noops
                                                    #   libero_10_no_noops

# ===== 一般不用改的参数 =====
VLA_PATH="openvla/openvla-7b"                       # 预训练 VLA 模型路径（HuggingFace）
BATCH_SIZE=8                                        # 每卡 batch size（8 需要 ~62GB 显存，1 只需 ~25GB）
LR=5e-4                                             # 学习率
DECAY_STEPS=100000                                  # LR 衰减前的步数
MAX_STEPS=150005                                    # 总训练步数
SAVE_FREQ=10000                                     # checkpoint 保存频率
LORA_RANK=32                                        # LoRA rank

# ===== 可选：Weights & Biases 日志 =====
USE_WANDB=False                                     # 是否启用 wandb
WANDB_ENTITY="YOUR_WANDB_ENTITY"                    # wandb 实体名
WANDB_PROJECT="YOUR_WANDB_PROJECT"                  # wandb 项目名

# ===== 运行训练 =====
torchrun --standalone --nnodes 1 --nproc-per-node ${NUM_GPUS} \
    vla-scripts/finetune.py \
    --vla_path ${VLA_PATH} \
    --data_root_dir ${DATA_ROOT} \
    --dataset_name ${DATASET} \
    --run_root_dir ${RUN_ROOT} \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --num_steps_before_decay ${DECAY_STEPS} \
    --max_steps ${MAX_STEPS} \
    --save_freq ${SAVE_FREQ} \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank ${LORA_RANK} \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --run_id_note "baseline--${DATASET}"
