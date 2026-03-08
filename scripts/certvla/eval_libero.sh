#!/bin/bash
# ============================================================================
# LIBERO 评估脚本
# ============================================================================
# 用途：在 LIBERO 仿真环境中评估训练好的模型
# 使用方法：
#   1. 修改下面的 ===== 用户必改参数 ===== 部分
#   2. bash scripts/certvla/eval_libero.sh
# 注意：评估需要 LIBERO 环境已安装（pip install -e LIBERO）
# ============================================================================

# ===== 用户必改参数 =====
CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial"  # 模型 checkpoint 路径
TASK_SUITE="libero_spatial"                                     # 评估任务集，可选：
                                                                #   libero_spatial
                                                                #   libero_object
                                                                #   libero_goal
                                                                #   libero_10

# ===== 一般不用改的参数 =====
NUM_TRIALS=50                # 每个任务的评估回合数
SEED=42                      # 随机种子

# ===== 可选参数 =====
USE_WANDB=False              # 是否使用 wandb 记录评估结果
WANDB_ENTITY=""
WANDB_PROJECT=""

# ===== 运行评估 =====
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${CHECKPOINT} \
    --task_suite_name ${TASK_SUITE} \
    --num_trials_per_task ${NUM_TRIALS} \
    --seed ${SEED} \
    --center_crop True \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --load_in_8bit False \
    --load_in_4bit False
