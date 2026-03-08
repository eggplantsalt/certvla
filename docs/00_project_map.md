# CertVLA 项目地图

> 本文档帮助你快速定位项目中每个模块的位置和职责。

## 项目概览

CertVLA（Certifiable Task State Abstraction）是在 **OpenVLA-OFT** 基础上开发的新型 VLA 训练与闭环执行框架。核心思想：VLA 的内部状态不应只是"记住更多历史"，而应是**对下一段动作的结构化后果可证实的最小充分状态**。

## 目录结构总览

```
openvla-oft/
│
├── certvla/                    ← ★ CertVLA 核心方法代码（我们新增的全部内容）
│   ├── slots/                  ← 任务状态 slot 定义与距离函数
│   ├── data/                   ← 数据标签层：chunk 样本、证书挖掘、目标签名
│   ├── model/                  ← 模型层：状态 token、readout、证书头、动作头
│   ├── training/               ← 训练层：7 个损失函数、课程调度、scheduled sampling
│   └── inference/              ← 推理层：证书 gap、repair 控制器、日志
│
├── prismatic/                  ← 上游 Prismatic VLM 框架（不要修改）
│   ├── models/vlas/openvla.py  ← OpenVLA 模型定义
│   ├── vla/datasets/           ← RLDS 数据集加载
│   └── vla/constants.py        ← ACTION_DIM=7, NUM_ACTIONS_CHUNK=8 等常量
│
├── vla-scripts/                ← 上游训练/部署脚本
│   ├── finetune.py             ← ★ 训练入口（49KB，未来 CertVLA 将在此集成）
│   └── deploy.py               ← 部署服务
│
├── experiments/robot/          ← 上游评估脚本
│   └── libero/
│       ├── run_libero_eval.py  ← ★ LIBERO 评估入口
│       └── libero_utils.py     ← 评估工具函数
│
├── configs/certvla/            ← CertVLA 配置
│   └── slots_v1.py             ← slot 词汇表默认参数
│
├── scripts/certvla/            ← ★ 运行脚本（shell scripts）
│   ├── train_baseline.sh       ← 基线训练
│   ├── eval_libero.sh          ← LIBERO 评估
│   ├── run_tests.sh            ← 单元测试
│   └── smoke_test.sh           ← 快速冒烟测试
│
├── tests/                      ← 单元测试（144 个测试）
│   ├── test_slot_schema.py     ← Phase 1: slot 词汇表测试
│   ├── test_slot_metrics.py    ← Phase 1: 距离函数测试
│   ├── test_certificate_mining.py ← Phase 1: 证书挖掘测试
│   ├── test_preserve_rules.py  ← Phase 1: preserve 规则测试
│   ├── test_goal_signature.py  ← Phase 1: 目标签名测试
│   ├── test_model_shapes.py    ← Phase 2: 模型 shape 和梯度测试
│   ├── test_losses.py          ← Phase 3: 损失函数和课程测试
│   └── test_inference.py       ← Phase 4: gap、repair、日志测试
│
└── docs/                       ← 文档（本目录）
```

## CertVLA 核心模块详解

### `certvla/slots/` — 任务状态 Slot 定义

| 文件 | 职责 |
|------|------|
| `schema.py` | 定义 10 个 v1 slot（SlotName 枚举、SlotDomain、SlotFamily、SlotMeta、SLOT_REGISTRY） |
| `metrics.py` | slot 距离函数 d_j(a,b)、值-张量转换、flat tensor 编码 |
| `role_sets.py` | 定义 J_E（5 个 enabling）、J_R（4 个 result）、J_C（1 个 confidence）、J_CERT |
| `preserve_rules.py` | 结构化 preserve 规则：latch-preserve、support-preserve |

### `certvla/data/` — 数据与标签层

| 文件 | 职责 |
|------|------|
| `chunk_sample.py` | 数据结构：SlotState、CertificateLabel、CertChunkSample |
| `certificate_mining.py` | 自动证书挖掘：advance（数据驱动）+ preserve（结构规则）+ ignore |
| `state_labels.py` | 状态标注接口：StateLabeler ABC、LiberoOracleLabeler（待实现） |
| `goal_signature.py` | 从 episode 末尾 K 步计算目标签名 s* |
| `counterfactuals.py` | 反事实样本构造接口（v1 占位） |
| `label_episodes.py` | 离线标注脚本框架 |

### `certvla/model/` — 模型层

| 文件 | 对应数学对象 | 职责 |
|------|-------------|------|
| `state_token.py` | z_t | 可学习初始状态 z_0 + 门控更新 z_t = g·z̃_t + (1-g)·z_{t-1} |
| `state_readout.py` | ŝ_t = R_φ(z_t) | 从 z_t 读出 10 个 slot 的预测值 |
| `certificate_head.py` | ĉ_t = Q_ψ(z_t) | 从 z_t 预测每个 cert slot 的 role 和 goal |
| `action_head.py` | Â_t | 证书条件动作头：coarse(z_t,c_t) + λ_res·fine(o_t,z_t,c_t) |
| `certvla_wrapper.py` | — | 组合以上所有头的 wrapper（不修改基座模型） |
| `outputs.py` | — | CertVLAOutput 数据类 |

### `certvla/training/` — 训练层

| 文件 | 职责 |
|------|------|
| `losses.py` | 7 个损失函数：L_state、L_role、L_goal、L_act、L_cons、L_dep、L_cf |
| `curriculum.py` | 4 阶段训练调度：state → certificate → policy → counterfactual |
| `sched_sampling.py` | Scheduled sampling：constant/linear/cosine 退火 |

### `certvla/inference/` — 推理层

| 文件 | 职责 |
|------|------|
| `gap.py` | 证书 gap 计算：per-slot gap γ_j + 聚合 gap Γ_t |
| `repair.py` | 短视野局部 repair 控制器（best-of-N 重试） |
| `logging.py` | 推理日志：StepRecord、EpisodeTrace、InferenceLogger |

## 上游代码（不要修改）

| 目录 | 说明 |
|------|------|
| `prismatic/` | Prismatic VLM 框架：视觉编码器、LLM backbone、投影层、数据加载 |
| `vla-scripts/` | OpenVLA-OFT 训练/部署脚本 |
| `experiments/` | 实验评估脚本（LIBERO、ALOHA） |
| `LIBERO/` | LIBERO 仿真环境（git 子模块） |

## 新人推荐阅读顺序

1. **本文档** (`docs/00_project_map.md`) — 了解项目结构
2. **`docs/02_theory_and_core_idea.md`** — 理解核心思想
3. **`certvla/slots/schema.py`** — 从 slot 定义开始看代码
4. **`certvla/model/certvla_wrapper.py`** — 理解整体前向流程
5. **`certvla/training/losses.py`** — 理解训练目标
6. **`certvla/inference/gap.py`** — 理解闭环推理
