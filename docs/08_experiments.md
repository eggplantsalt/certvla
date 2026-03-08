# 实验配置指南

本文档介绍如何在 LIBERO 基准上配置、运行和分析 CertVLA / OpenVLA-OFT 实验。

---

## 1. LIBERO 基准简介

LIBERO 是一个面向桌面机器人操作的仿真基准平台，包含 **5 个任务集**，从不同维度评估策略的泛化能力：

| 任务集 | 任务数 | 评估重点 | 说明 |
|--------|--------|----------|------|
| **LIBERO-Spatial** | 10 | 空间泛化 | 相同物体、不同空间布局 |
| **LIBERO-Object** | 10 | 物体泛化 | 相同布局、不同目标物体 |
| **LIBERO-Goal** | 10 | 目标泛化 | 相同场景、不同任务目标 |
| **LIBERO-Long** | 10 | 长程任务 | 需要多步骤完成的长序列任务 |
| **LIBERO-10** | 10 | 综合能力 | 从以上任务集中抽取的混合集 |

在训练脚本中，数据集名称使用 `_no_noops` 后缀（已去除空操作帧），对应关系如下：

```
libero_spatial_no_noops   →  LIBERO-Spatial
libero_object_no_noops    →  LIBERO-Object
libero_goal_no_noops      →  LIBERO-Goal
libero_10_no_noops        →  LIBERO-10
```

数据格式为 RLDS（Record-oriented Dataset for Learned Skills），需要预先转换并存放在指定的 `DATA_ROOT` 目录下。

---

## 2. OpenVLA-OFT 基线配置

### 2.1 训练配置

基线训练脚本位于 `scripts/certvla/train_baseline.sh`，核心配置如下：

```bash
# === 用户必改参数 ===
NUM_GPUS=1                                  # GPU 数量
DATA_ROOT="/PATH/TO/modified_libero_rlds"   # RLDS 数据集根目录
RUN_ROOT="/PATH/TO/checkpoints"             # checkpoint 和日志输出目录
DATASET="libero_spatial_no_noops"           # 数据集名称

# === 模型与训练参数 ===
VLA_PATH="openvla/openvla-7b"              # 预训练 VLA 模型 (HuggingFace)
BATCH_SIZE=8                                # 每卡 batch size
LR=5e-4                                     # 学习率
LORA_RANK=32                                # LoRA rank
MAX_STEPS=150005                            # 总训练步数
DECAY_STEPS=100000                          # LR 衰减前的步数
SAVE_FREQ=10000                             # checkpoint 保存频率
```

训练使用 `torchrun` 启动分布式训练：

```bash
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
    --lora_rank ${LORA_RANK} \
    --image_aug True \
    ...
```

### 2.2 评估配置

评估脚本位于 `scripts/certvla/eval_libero.sh`，核心配置如下：

```bash
# === 用户必改参数 ===
CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial"  # 模型 checkpoint
TASK_SUITE="libero_spatial"                                     # 评估任务集

# === 评估参数 ===
NUM_TRIALS=50          # 每个任务的评估回合数
SEED=42                # 随机种子
```

评估通过 `experiments/robot/libero/run_libero_eval.py` 执行：

```bash
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${CHECKPOINT} \
    --task_suite_name ${TASK_SUITE} \
    --num_trials_per_task ${NUM_TRIALS} \
    --seed ${SEED} \
    --center_crop True \
    --use_l1_regression True \
    --num_images_in_input 2 \
    --use_proprio True \
    --load_in_8bit False \
    --load_in_4bit False
```

**重要提示**：评估需要安装 LIBERO 仿真环境 (`pip install -e LIBERO`)。

---

## 3. 关键训练超参数

### 3.1 核心超参数汇总

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 8 | 每卡 batch size；8 需要约 62GB 显存，1 仅需约 25GB |
| `learning_rate` | 5e-4 | 基础学习率，搭配线性衰减调度器 |
| `lora_rank` | 32 | LoRA 低秩适配的秩参数 |
| `action_dim` | 7 | 动作空间维度（6 关节 + 1 抓手） |
| `chunk_size` (num_actions_chunk) | 8 | 每次预测的动作块长度 |
| `max_steps` | 150005 | 总训练步数 |
| `num_steps_before_decay` | 100000 | 学习率开始衰减的步数 |
| `save_freq` | 10000 | checkpoint 保存间隔 |
| `num_images_in_input` | 2 | 输入图像数量（当前帧 + 历史帧） |
| `image_aug` | True | 是否启用图像增强 |

### 3.2 CertVLA 特有超参数

CertVLA 在基线之上增加了以下关键超参数：

**损失函数权重** (`certvla/training/curriculum.py`)：

| 权重参数 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | 对应损失项 |
|----------|---------|---------|---------|---------|------------|
| `lambda_s` | 1.0 | 1.0 | 1.0 | 1.0 | L_state（状态读出） |
| `lambda_r` | 0.0 | 1.0 | 1.0 | 1.0 | L_role（角色分类） |
| `lambda_g` | 0.0 | 1.0 | 1.0 | 1.0 | L_goal（目标预测） |
| `lambda_a` | 0.0 | 0.0 | 1.0 | 1.0 | L_act（动作回归） |
| `lambda_c` | 0.0 | 0.0 | 0.5 | 0.5 | L_cons（一致性） |
| `lambda_d` | 0.0 | 0.0 | 0.5 | 0.5 | L_dep（依赖性） |
| `lambda_cf` | 0.0 | 0.0 | 0.0 | 0.5 | L_cf（反事实） |

**课程学习阶段边界**（默认步数范围）：

| 阶段 | 步数范围 | 启用损失项 | 冻结模块 |
|------|----------|-----------|----------|
| Stage 1 (state) | 0 -- 5,000 | L_state | backbone, certificate, action |
| Stage 2 (certificate) | 5,000 -- 15,000 | + L_role, L_goal | backbone, action |
| Stage 3 (policy) | 15,000 -- 40,000 | + L_act, L_cons, L_dep | backbone |
| Stage 4 (counterfactual) | 40,000 -- 60,000 | + L_cf | backbone |

**其他 CertVLA 超参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `focal_gamma` | 2.0 | L_role 的 Focal CE 聚焦参数 |
| `dep_margin` | 0.1 | L_dep 的三元组边距 |
| `lambda_pre` | 1.0 | L_cons 中 preserve 项的权重 |
| `cf_mu` | 1.0 | L_cf breaking 项的边距 |
| `gap_threshold` | 0.3 | 推理时触发 repair 的间隙阈值 |
| `max_repair_steps` | 3 | 最大 repair 重试次数 |
| `lambda_res_init` | 0.1 | 动作头残差缩放因子初始值 |

---

## 4. GPU 需求

### 4.1 单卡配置

| 配置 | 显存需求 | 说明 |
|------|----------|------|
| `batch_size=1` | 约 25GB | 最小配置，可在 RTX 3090/4090 上运行 |
| `batch_size=8` | 约 62GB | 推荐配置，需要 A100 80GB 或 H100 |
| `batch_size=8` + `load_in_8bit` | 约 40GB | 8-bit 量化训练，A100 40GB 可运行 |

### 4.2 多卡配置

训练脚本使用 `torchrun` 支持数据并行。修改 `NUM_GPUS` 即可：

```bash
NUM_GPUS=4  # 使用 4 张 GPU
```

**等效 batch size 计算**：
```
全局 batch size = NUM_GPUS x batch_size
```

例如 4 卡 x `batch_size=8` = 全局 batch size 32。增大全局 batch size 后建议等比例调整学习率：

```
调整后学习率 = 基础学习率 x (全局 batch size / 8)
```

### 4.3 评估时 GPU 需求

评估（推理）的显存需求远低于训练：

- 不量化：约 16--20GB
- 8-bit 量化 (`--load_in_8bit True`)：约 10GB
- 4-bit 量化 (`--load_in_4bit True`)：约 8GB

CertVLA 附加模块（StateReadout、CertificateHead、CertActionHead）增加的参数量较小（约几十 MB），对显存影响可忽略。

### 4.4 纯 CPU 测试

CertVLA 的所有单元测试（144 个）不需要 GPU，纯 CPU 即可运行：

```bash
bash scripts/certvla/smoke_test.sh
```

---

## 5. 评估指标

### 5.1 主要指标：成功率 (Success Rate)

LIBERO 评估的核心指标是 **任务成功率**：

```
成功率 = 成功回合数 / 总回合数
```

每个任务独立评估（默认 `NUM_TRIALS=50` 个回合），每个任务集包含 10 个任务。

### 5.2 评估报告结构

评估结果按以下层级组织：

```
任务集级别:
  LIBERO-Spatial 整体成功率: XX.X%

  单任务级别:
    Task 0: XX.X% (25/50)
    Task 1: XX.X% (30/50)
    ...
    Task 9: XX.X% (40/50)
```

### 5.3 CertVLA 附加评估指标

除成功率外，CertVLA 还可输出以下诊断指标（通过 `InferenceLogger` 记录）：

| 指标 | 说明 |
|------|------|
| `mean_gap` | 平均证书间隙 (Gamma_t) |
| `max_gap` | 最大证书间隙 |
| `num_repairs` | repair 触发次数 |
| `total_attempts` | 总前向传播次数（含 repair 重试） |
| `gap_history` | 每步间隙值的时间序列 |
| `role_probs` | 各槽位角色概率分布 |
| `gate_value` | 状态令牌门控值均值（用于监控状态更新幅度） |

---

## 6. 消融实验设计建议

### 6.1 核心消融：CertVLA 模块对比

| 实验编号 | 条件 | 说明 |
|----------|------|------|
| A1 | OpenVLA-OFT 基线 (无 CertVLA) | 使用 `train_baseline.sh`，纯 L1 回归动作头 |
| A2 | CertVLA (完整) | 全部 4 个 Stage，含 repair |
| A3 | CertVLA (无 repair) | `gap_threshold=999.0`（永不触发 repair） |
| A4 | CertVLA (无证书条件化) | 只用 state readout，不用 certificate head 条件化动作 |

### 6.2 Repair 机制消融

| 实验编号 | gap_threshold | max_repair_steps | 说明 |
|----------|---------------|-----------------|------|
| R1 | 0.1 | 3 | 低阈值（积极 repair） |
| R2 | 0.3 | 3 | 默认配置 |
| R3 | 0.5 | 3 | 高阈值（保守 repair） |
| R4 | 0.3 | 1 | 最多 1 次 repair |
| R5 | 0.3 | 5 | 最多 5 次 repair |

### 6.3 训练阶段消融

| 实验编号 | 训练范围 | 说明 |
|----------|----------|------|
| S1 | 仅 Stage 1 | 只训练状态读出 |
| S2 | Stage 1--2 | 状态读出 + 证书头 |
| S3 | Stage 1--3 | 不含反事实（推荐的最小完整配置） |
| S4 | Stage 1--4 | 完整训练（含反事实增强） |

### 6.4 Gap 阈值灵敏度分析

建议在评估时扫描不同的 gap_threshold 值，观察成功率与 repair 频率的权衡：

```python
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
```

对于每个阈值，记录：
- 成功率
- 平均 repair 次数
- 平均推理延迟（repair 增加前向传播次数）

### 6.5 槽位重要性消融

通过修改 `slot_weights` 参数，可以评估各个槽位对 gap 计算的贡献：

```python
# 逐一置零某个槽位的权重
slot_weights = {SlotName.EE_TARGET_PROXIMITY: 0.0}  # 禁用该槽位
```

---

## 7. 实验结果记录

### 7.1 Weights & Biases (wandb) 集成

训练脚本内置 wandb 支持。在 `train_baseline.sh` 中启用：

```bash
USE_WANDB=True
WANDB_ENTITY="your-team-name"
WANDB_PROJECT="certvla-experiments"
```

wandb 会自动记录以下内容：
- 训练损失曲线（总损失及各分量）
- 学习率调度
- 梯度统计

**CertVLA 建议额外记录的 wandb 自定义指标**：

```python
import wandb

# 在训练循环中记录各损失分量
wandb.log({
    "loss/total": total_loss,
    "loss/state": components["state"],
    "loss/role": components["role"],
    "loss/goal": components["goal"],
    "loss/action": components["action"],
    "loss/consistency": components["consistency"],
    "loss/dependence": components["dependence"],
    "loss/counterfactual": components["counterfactual"],
    "curriculum/stage": current_stage.value,
    "diagnostics/gate_mean": gate.mean().item(),
    "diagnostics/lambda_res": model.cert_action_head.lambda_res.item(),
    "step": global_step,
})
```

### 7.2 TensorBoard 集成

如果不使用 wandb，可以使用 TensorBoard：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f"{RUN_ROOT}/tensorboard")

# 记录损失分量
for name, value in components.items():
    writer.add_scalar(f"loss/{name}", value, global_step)

# 记录课程阶段
writer.add_scalar("curriculum/stage_id", stage_id, global_step)

# 记录门控值
writer.add_scalar("diagnostics/gate_mean", gate.mean().item(), global_step)

writer.close()
```

### 7.3 InferenceLogger 评估记录

CertVLA 提供了 `InferenceLogger` 类（位于 `certvla/inference/logging.py`）用于评估时的详细记录：

```python
from certvla.inference.logging import InferenceLogger

logger = InferenceLogger(verbose=True, max_episodes=100)

# 评估循环
logger.begin_episode(metadata={"task": task_name, "trial": trial_id})

for step in range(max_steps):
    actions, gap, n_repairs = controller.step(...)
    # StepRecord 由 RepairController 内部自动记录

logger.end_episode()

# 获取统计摘要
trace = logger.get_last_trace()
summary = trace.summary()
# summary = {
#     "num_steps": ...,
#     "num_repairs": ...,
#     "total_attempts": ...,
#     "mean_gap": ...,
#     "max_gap": ...,
#     "num_warnings": ...,
# }
```

### 7.4 建议的实验结果表格格式

```
| 方法 | Spatial | Object | Goal | Long | 10 | 平均 |
|------|---------|--------|------|------|----|------|
| OpenVLA-OFT (基线) | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X |
| CertVLA (无 repair) | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X |
| CertVLA (完整)       | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X |
```

每个单元格应报告 **平均成功率 +/- 标准差**，建议至少使用 3 个不同的随机种子。

---

## 8. 快速上手：完整实验流程

```bash
# 步骤 1: 运行 smoke test 验证环境
bash scripts/certvla/smoke_test.sh

# 步骤 2: 训练基线模型
# 修改 train_baseline.sh 中的 DATA_ROOT 和 RUN_ROOT 后：
bash scripts/certvla/train_baseline.sh

# 步骤 3: 评估基线模型
# 修改 eval_libero.sh 中的 CHECKPOINT 后：
bash scripts/certvla/eval_libero.sh

# 步骤 4: 运行特定模块的单元测试
bash scripts/certvla/run_tests.sh losses      # 损失函数测试
bash scripts/certvla/run_tests.sh inference    # 推理测试
bash scripts/certvla/run_tests.sh model        # 模型测试
bash scripts/certvla/run_tests.sh data         # 数据层测试
```
