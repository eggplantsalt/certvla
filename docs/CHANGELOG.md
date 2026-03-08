# CertVLA 变更日志

> 本文件记录 CertVLA 模块在 OpenVLA-OFT 仓库中的所有新增和修改。
> 上游代码（prismatic/、vla-scripts/、experiments/）无任何修改。

---

## Phase 1: 数据与标注层

**新增文件（14 个源文件 + 5 个测试文件 + 1 个配置）：**

| 文件 | 说明 |
|------|------|
| `certvla/__init__.py` | 包根模块 |
| `certvla/slots/__init__.py` | Slot 子包，重导出核心 API |
| `certvla/slots/schema.py` | v1 冻结词汇表：10 个 slot 定义、SlotName/SlotDomain/SlotFamily 枚举、SlotMeta 数据类、SLOT_REGISTRY |
| `certvla/slots/metrics.py` | 逐 slot 距离函数 `d_j(a, b)`、值↔张量转换、flat tensor 拼接（dim=16） |
| `certvla/slots/role_sets.py` | J_E (5)、J_R (4)、J_C (1)、J_CERT (9) 族集合，附编译期断言 |
| `certvla/slots/preserve_rules.py` | latch-preserve、support-preserve（5 条结构化规则）、compute_preserve_set |
| `certvla/data/__init__.py` | 数据子包 |
| `certvla/data/chunk_sample.py` | SlotState、CertificateLabel、CertChunkSample 数据类 |
| `certvla/data/state_labels.py` | StateLabeler (ABC)、PseudoLabelInterface (ABC)、LiberoOracleLabeler 接口 |
| `certvla/data/certificate_mining.py` | `mine_certificate()` —— advance/preserve/ignore 挖掘算法，4 个度量指标 |
| `certvla/data/goal_signature.py` | `compute_goal_signature()` —— 回合级目标签名（均值/投票/众数聚合） |
| `certvla/data/label_episodes.py` | 离线标注脚本结构、save/load 接口 |
| `certvla/data/counterfactuals.py` | CounterfactualPair、CounterfactualBuilder (ABC) |
| `configs/certvla/slots_v1.py` | 默认挖掘阈值、目标 K、chunk 大小 |
| `tests/conftest.py` | 共享 fixture（slot_registry、make_slot_state） |
| `tests/test_slot_schema.py` | 20 个 slot schema 测试 |
| `tests/test_slot_metrics.py` | 18 个距离/张量测试 |
| `tests/test_certificate_mining.py` | 9 个证书挖掘测试（含 4-chunk 场景端到端） |
| `tests/test_preserve_rules.py` | 11 个 preserve 规则测试 |
| `tests/test_goal_signature.py` | 9 个目标签名测试 |

**测试结果：74 个通过**

---

## Phase 2: 模型层

**新增文件（7 个源文件 + 2 个测试文件）：**

| 文件 | 说明 |
|------|------|
| `certvla/model/__init__.py` | 模型子包 |
| `certvla/model/outputs.py` | CertVLAOutput 数据类（前向传播统一输出） |
| `certvla/model/state_token.py` | StateTokenModule —— z_0 可学习初始状态 + 门控更新 |
| `certvla/model/state_readout.py` | StateReadoutHead —— 共享 trunk + 10 个逐 slot 输出头 |
| `certvla/model/certificate_head.py` | CertificateHead —— 共享 trunk + 9 个 role_head + goal_head |
| `certvla/model/action_head.py` | CertActionHead —— CertificateEmbedding + CoarseActionBranch + FineActionBranch |
| `certvla/model/certvla_wrapper.py` | CertVLAWrapper —— 组合所有子模块的 6 步前向流程 |
| `tests/test_model.py` | 21 个模型层测试（含端到端前向 + 梯度流验证） |

**阶段测试结果：95 个通过（Phase 1 + 2）**

---

## Phase 3: 训练目标与阶段训练框架

**新增文件（4 个源文件 + 1 个测试文件）：**

| 文件 | 说明 |
|------|------|
| `certvla/training/__init__.py` | 训练子包，重导出所有损失函数、课程类、采样器 |
| `certvla/training/losses.py` | 7 个损失函数：L_state / L_role / L_goal / L_act / L_cons / L_dep / L_cf + cert_total_loss 加权组合 |
| `certvla/training/curriculum.py` | TrainingStage 枚举、StageConfig 数据类、DEFAULT_STAGES (4 个内置阶段)、CurriculumScheduler |
| `certvla/training/sched_sampling.py` | ScheduledSampler —— 3 种退火策略 (constant / linear / cosine) |
| `tests/test_losses.py` | 30 个训练测试（含 7 个损失函数 + 课程 + 采样 + 端到端梯度流） |

**损失函数详情：**

| 简称 | 函数名 | 用途 | 首次激活 |
|------|--------|------|---------|
| L_state | `cert_state_loss` | 逐 slot 状态读出 | Stage 1 |
| L_role | `cert_role_loss` | 角色分类 (focal CE γ=2.0) | Stage 2 |
| L_goal | `cert_goal_loss` | advance 目标值 | Stage 2 |
| L_act | `cert_action_loss` | L1 动作回归 | Stage 3 |
| L_cons | `cert_consistency_loss` | advance + preserve 一致性 | Stage 3 |
| L_dep | `cert_dependence_loss` | margin triplet 依赖性 | Stage 3 |
| L_cf | `cert_counterfactual_loss` | 不变性 + 破坏性 | Stage 4 |

**阶段测试结果：125 个通过（Phase 1 + 2 + 3）**

---

## Phase 4: 推理闭环与修复

**新增文件（4 个源文件 + 1 个测试文件）：**

| 文件 | 说明 |
|------|------|
| `certvla/inference/__init__.py` | 推理子包 |
| `certvla/inference/gap.py` | slot_gap —— 逐 slot 证书间隙、aggregate_certificate_gap —— 聚合间隙、GapResult 数据类 |
| `certvla/inference/repair.py` | RepairConfig + RepairController —— best-of-N 短程修复循环 |
| `certvla/inference/logging.py` | StepRecord / EpisodeTrace / InferenceLogger —— 推理调试日志 |
| `tests/test_inference.py` | 19 个推理测试（含 5 步 fake rollout 端到端） |

**阶段测试结果：144 个通过（全部 Phase 1-4）**

---

## 工程化整理

### Shell 脚本

| 文件 | 说明 |
|------|------|
| `scripts/certvla/train_baseline.sh` | OpenVLA-OFT 基线训练（torchrun 多卡、参数化配置） |
| `scripts/certvla/eval_libero.sh` | LIBERO 评估脚本 |
| `scripts/certvla/run_tests.sh` | 单元测试执行器（支持按模块选择：data / model / losses / inference） |
| `scripts/certvla/smoke_test.sh` | 冒烟测试（包导入验证 + 全量 pytest） |

### 中文代码注释

对 `certvla/` 下全部 ~28 个 Python 源文件添加了详细的中文注释：

- **certvla/slots/** (4 个文件)：模块头说明、设计决策、slot 语义、常见陷阱
- **certvla/model/** (6 个文件)：模块架构、张量形状流、门控初始化策略
- **certvla/data/** (6 个文件)：挖掘算法逐步注释、度量指标公式、数据流
- **certvla/training/** (3 个文件)：损失函数公式推导、课程学习策略、采样接口
- **certvla/inference/** (3 个文件)：间隙计算公式、修复循环逻辑、日志结构

### 中文文档

| 文件 | 主题 |
|------|------|
| `docs/00_project_map.md` | 项目目录导览与推荐阅读顺序 |
| `docs/01_quickstart.md` | 从零开始的安装验证指南 |
| `docs/02_theory_and_core_idea.md` | 核心数学对象与代码对照表 |
| `docs/03_code_structure.md` | 代码结构详解 |
| `docs/04_data_pipeline.md` | 数据管线与证书挖掘 |
| `docs/05_state_and_certificate.md` | 状态与证书机制深入解析 |
| `docs/06_training_pipeline.md` | 训练流水线、损失函数、课程学习 |
| `docs/07_inference_and_repair.md` | 推理闭环与修复机制 |
| `docs/08_experiments.md` | 实验配置指南 |
| `docs/09_troubleshooting.md` | 常见问题排查 |
| `docs/CHANGELOG.md` | 本文件 |

---

## 统计摘要

| 指标 | 数量 |
|------|------|
| CertVLA Python 源文件 | 28 |
| 测试文件 | 7 |
| 测试用例 | 144 |
| Shell 脚本 | 4 |
| 中文文档 | 11 |
| 上游文件修改 | 0 |

---

## 未修改文件声明

以下文件为**只读参考**，未做任何修改：
- `docs/CONTEXT.md` — 原始上下文文档
- `docs/certvla_code_agent_context.md` — 代码代理上下文
- `prismatic/` — 上游模型代码
- `vla-scripts/` — 上游训练/评估脚本
- `experiments/` — 上游实验配置
