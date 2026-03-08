# 常见问题排查

本文档汇总了 CertVLA / OpenVLA-OFT 开发和使用过程中的常见问题及解决方案。

---

## 安装问题

### `draccus` 安装失败或版本不兼容

**现象**：`pip install` 时报错找不到 `draccus`，或运行时出现 `AttributeError`。

**原因**：`draccus` 是 OpenVLA-OFT 使用的配置解析库，尚未发布到 PyPI 主仓库，需要从 GitHub 源码安装。

**解决方案**：

```bash
pip install git+https://github.com/dlwh/draccus.git
```

如果仍有版本冲突，尝试锁定特定 commit：

```bash
pip install git+https://github.com/dlwh/draccus.git@<commit-hash>
```

---

### TensorFlow 依赖冲突

**现象**：安装 OpenVLA-OFT 依赖时出现 TensorFlow 版本冲突，或运行时出现 `tf.io` 相关错误。

**原因**：RLDS 数据加载依赖 TensorFlow Datasets，但 TF 与 PyTorch 的 CUDA 版本可能冲突。

**解决方案**：

1. 安装 CPU 版 TensorFlow（仅用于数据加载）：
   ```bash
   pip install tensorflow-cpu
   ```

2. 如果不需要训练数据加载（仅运行 CertVLA 单元测试），可以完全跳过 TF：
   ```bash
   # CertVLA 模块不依赖 TensorFlow
   bash scripts/certvla/smoke_test.sh  # 无需 TF
   ```

---

### `flash_attn` 安装失败

**现象**：`pip install flash-attn` 编译失败，报 CUDA 版本或编译器错误。

**原因**：`flash_attn` 需要从源码编译，对 CUDA 版本、GCC 版本和 GPU 架构有严格要求。

**解决方案**：

1. 确认 CUDA 版本匹配：
   ```bash
   nvcc --version         # 系统 CUDA
   python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
   ```

2. 使用预编译 wheel（推荐）：
   ```bash
   pip install flash-attn --no-build-isolation
   ```

3. 如果始终无法安装，可以禁用 flash attention（会降低训练速度但不影响正确性）：
   - 在模型配置中设置 `attn_implementation="eager"` 或 `attn_implementation="sdpa"`

4. Windows 环境下的特殊注意事项：
   - `flash_attn` 对 Windows 的支持有限，建议使用 WSL2 或 Docker
   - 替代方案：使用 PyTorch 内置的 `torch.nn.functional.scaled_dot_product_attention`

---

## 测试问题

### 如何运行特定模块的测试

CertVLA 提供了便捷的测试脚本 `scripts/certvla/run_tests.sh`：

```bash
# 运行全部测试（144 个）
bash scripts/certvla/run_tests.sh

# 按模块运行
bash scripts/certvla/run_tests.sh data         # Phase 1 数据层测试
bash scripts/certvla/run_tests.sh model        # Phase 2 模型测试
bash scripts/certvla/run_tests.sh losses       # Phase 3 损失函数测试
bash scripts/certvla/run_tests.sh inference    # Phase 4 推理测试

# 按关键字运行（pytest -k 过滤）
bash scripts/certvla/run_tests.sh "slot"       # 所有包含 "slot" 的测试
bash scripts/certvla/run_tests.sh "gap"        # 所有包含 "gap" 的测试
```

也可以直接使用 pytest：

```bash
# 运行单个测试文件
python -m pytest tests/test_losses.py -v --tb=short

# 运行单个测试函数
python -m pytest tests/test_inference.py::test_repair_triggers -v

# 显示详细输出
python -m pytest tests/ -v --tb=long -s
```

---

### 常见 UserWarning 告警

**现象**：运行测试时出现大量 `UserWarning` 信息。

**常见告警及含义**：

1. **`UserWarning: torch.nn.utils.weight_norm is deprecated`**
   - 来源：某些旧版模块使用了 `weight_norm`
   - 影响：无，纯警告信息
   - 抑制方法：
     ```python
     import warnings
     warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.utils")
     ```

2. **`UserWarning: Plan failed with a cudnnException`**
   - 来源：cuDNN 的某些算法在当前硬件上不可用
   - 影响：PyTorch 会自动回退到其他算法
   - 抑制方法：可忽略

3. **`UserWarning: TypedStorage is deprecated`**
   - 来源：PyTorch 2.x 内部 API 变更
   - 影响：无
   - 抑制方法：升级 PyTorch 到最新版本

---

## 训练问题

### CUDA OOM（显存不足）

**现象**：`RuntimeError: CUDA out of memory`

**解决方案**（按优先级排序）：

1. **降低 batch size**：
   ```bash
   BATCH_SIZE=1  # 从 8 降至 1，显存需求从 ~62GB 降至 ~25GB
   ```

2. **启用梯度检查点 (gradient checkpointing)**：
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **使用 8-bit 或 4-bit 量化训练**：
   ```bash
   --load_in_8bit True   # 减少约 40% 显存
   --load_in_4bit True   # 减少约 60% 显存
   ```

4. **减少输入图像数量**：
   ```bash
   --num_images_in_input 1  # 从 2 降至 1
   ```

5. **冻结更多模块**：CertVLA 的课程学习已经在各阶段冻结了不需要训练的模块，但可以进一步冻结 backbone：
   ```python
   freeze_backbone=True  # 默认已开启
   ```

6. **多卡分布式训练**：增加 `NUM_GPUS` 以分摊显存。

---

### 梯度爆炸 (Gradient Explosion)

**现象**：loss 突然变为 `NaN` 或 `Inf`，训练崩溃。

**诊断方法**：

```python
# 在训练循环中添加梯度监控
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100:
            print(f"WARNING: {name} grad_norm = {grad_norm}")
```

**解决方案**：

1. **降低学习率**：
   ```bash
   LR=1e-4  # 从 5e-4 降至 1e-4
   ```

2. **启用梯度裁剪**：
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **检查损失权重**：CertVLA 的 7 个损失项如果权重设置不当，某些项可能产生异常大的梯度。检查 `cert_total_loss` 返回的 `components` 字典，找到异常大的分量：
   ```python
   total, components = cert_total_loss(losses, weights)
   for name, val in components.items():
       print(f"  {name}: {val:.6f}")
   ```

4. **检查 Focal CE 的 gamma 值**：过大的 `focal_gamma` 可能导致数值不稳定。建议保持 `gamma=2.0`。

---

### Loss 不下降

**现象**：训练 loss 停滞不降或剧烈振荡。

**排查步骤**：

1. **检查课程阶段是否正确切换**：
   ```python
   scheduler = CurriculumScheduler()
   config = scheduler.get_config(current_step)
   print(f"Step {current_step}: stage={config.stage}, weights={config.loss_weights()}")
   ```
   确认当前 step 对应的 stage 和损失权重符合预期。

2. **分别查看各损失分量**：某个分量下降但另一个上升可能导致总 loss 看起来不动。使用 `cert_total_loss` 返回的 `components` 逐项检查。

3. **检查数据加载**：
   - 确认 `DATA_ROOT` 路径正确
   - 确认数据集名称拼写正确（注意 `_no_noops` 后缀）
   - 检查数据是否被正确 shuffle

4. **检查模块冻结状态**：如果需要训练的模块被意外冻结，loss 自然不会下降：
   ```python
   for name, param in model.named_parameters():
       if param.requires_grad:
           print(f"TRAINABLE: {name}")
   ```

5. **确认学习率不为零**：检查学习率调度器是否已衰减到极小值。

---

## 推理问题

### Gap 始终为 0（proxy gap 问题）

**现象**：推理时 `GapResult.aggregated` 始终为 0 或极小值。

**原因**：当 `state_readout_tH`（t+H 时刻的状态读出）未提供时，`RepairController._compute_gap` 使用 `goal_preds` 作为代理。此时会出现：

```python
# 代理计算逻辑 (certvla/inference/repair.py, _compute_gap):
if state_readout_tH is None:
    state_readout_tH = output.goal_preds
    # 这导致: d(goal_preds, goal_preds) = 0（advance 项）
    # 以及: d(state_readout, goal_preds) 可能也很小
```

advance 项 `d_j(goal^j, state_readout_tH^j) = d_j(goal^j, goal^j) = 0` 恒为零，导致整体 gap 被严重低估。

**解决方案**：

1. **提供真实的 `state_readout_tH`**：在闭环评估中，可以在执行完 action chunk 后做一次额外的前向传播，获取 t+H 时刻的状态读出。

2. **使用两步前向策略**：
   ```python
   # 第一步：正常推理，获取动作
   output_t = model(observation_t)
   # 执行动作...
   # 第二步：观测 t+H，前向传播获取 state_readout
   output_tH = model(observation_tH)
   state_readout_tH = output_tH.state_readout
   # 用真实 state_readout_tH 计算 gap
   gap = controller._compute_gap(output_t, state_readout_tH)
   ```

3. **暂时使用自一致性 gap**：如果无法获取 t+H 观测，可以接受 proxy gap 的局限性，但需要意识到此时 repair 机制基本不会触发。

---

### Repair 总是触发（threshold 太低）

**现象**：每一步推理都触发 repair，导致推理速度慢 3--4 倍。

**原因**：`gap_threshold` 设置过低，或 gap 计算本身产生了偏高的值。

**诊断方法**：

```python
logger = InferenceLogger(verbose=True)
# 运行若干 episode 后检查
for trace in logger.get_all_traces():
    gaps = trace.gap_history
    print(f"Gap 统计: mean={sum(gaps)/len(gaps):.4f}, "
          f"max={max(gaps):.4f}, min={min(gaps):.4f}")
    print(f"Repair 比例: {trace.num_repairs}/{trace.num_steps}")
```

**解决方案**：

1. **提高 gap_threshold**：
   ```python
   config = RepairConfig(gap_threshold=0.5)  # 默认 0.3，调高到 0.5
   ```

2. **检查 slot_weights**：某些槽位可能一直输出高 gap。通过检查 `GapResult.per_slot` 定位问题槽位：
   ```python
   for slot, gap_val in gap_result.per_slot.items():
       print(f"  {slot.value}: {gap_val.mean().item():.4f}")
   ```

3. **降低 max_repair_steps**：即使 repair 触发，限制重试次数也能控制延迟：
   ```python
   config = RepairConfig(max_repair_steps=1)
   ```

4. **完全禁用 repair**（用于对比实验）：
   ```python
   config = RepairConfig(gap_threshold=999.0)  # 永不触发
   ```

---

## 导入问题

### `certvla` 模块导入失败

**现象**：`ModuleNotFoundError: No module named 'certvla'`

**解决方案**：

确保从项目根目录运行，或将项目根目录添加到 Python 路径：

```bash
# 方法 1：从项目根目录运行
cd /path/to/openvla-oft
python -c "import certvla; print('OK')"

# 方法 2：设置 PYTHONPATH
export PYTHONPATH=/path/to/openvla-oft:$PYTHONPATH

# 方法 3：以开发模式安装
pip install -e /path/to/openvla-oft
```

---

### `certvla` 模块与 `prismatic` 的隔离

**现象**：导入 `certvla` 时触发 `prismatic` 相关依赖链的导入错误。

**设计说明**：`certvla` 包被设计为 **完全独立于 `prismatic`**。`certvla/` 目录下的所有文件均不导入 `prismatic` 模块。这是刻意的设计决策，目的是：

1. 避免触发 `prismatic` 的重量级依赖链（TensorFlow、大型视觉模型等）
2. 确保单元测试可以在轻量级环境中运行
3. 保持模块边界清晰

**如果遇到导入问题**：

- 检查是否在 `certvla/` 包内意外添加了对 `prismatic` 的导入
- `certvla/model/action_head.py` 中使用了硬编码默认值（`_DEFAULT_ACTION_DIM=7`、`_DEFAULT_NUM_ACTIONS_CHUNK=8`）而非从 `prismatic` 导入
- 集成层（将 CertVLA 嵌入 OpenVLA 流水线的代码）应在 `certvla/` 包外部实现

---

## 数据问题

### Slot 值超出范围

**现象**：`SlotMeta.validate_value()` 返回 `False`，或训练时出现 `NaN` 损失。

**合法值范围**：

| Slot 类型 | 值域 | 示例 |
|-----------|------|------|
| BINARY | `{0, 1}` 或 `{0.0, 1.0}` | `target_contact: 1` |
| CONTINUOUS | `[0.0, 1.0]` | `ee_target_proximity: 0.35` |
| CONFIDENCE | `[0.0, 1.0]` | `task_visible_confidence: 0.9` |
| CATEGORICAL | 预定义标签集 | `hand_occupancy: "target"` |

**检查方法**：

```python
from certvla.slots.schema import SLOT_REGISTRY, SlotName

meta = SLOT_REGISTRY[SlotName.EE_TARGET_PROXIMITY]
print(meta.validate_value(0.5))   # True
print(meta.validate_value(1.5))   # False - 超出 [0, 1]
print(meta.validate_value(-0.1))  # False - 超出 [0, 1]
```

**常见错误**：

- 忘记将连续值归一化到 `[0, 1]`
- 二值槽位传入了浮点值（如 `0.7`），应为 `0` 或 `1`
- 置信度槽位传入了负值

---

### Categorical 标签拼写错误

**现象**：`validate_value()` 返回 `False`，或训练时 `F.cross_entropy` 报 target 越界。

**原因**：分类槽位要求标签必须是预定义的合法字符串之一。

**合法标签列表**（v1 schema）：

| Slot | 合法标签 |
|------|----------|
| `hand_occupancy` | `"empty"`, `"target"`, `"other"` |
| `support_relation` | `"none"`, `"on_goal"`, `"on_other"` |
| `containment_relation` | `"none"`, `"in_goal"`, `"in_other"` |

**常见拼写错误**：

```python
# 错误示例
"Empty"      # 应为 "empty"（全小写）
"on goal"    # 应为 "on_goal"（使用下划线）
"in_target"  # 应为 "in_goal"（使用正确标签名）
"grasped"    # 应为 "target"（hand_occupancy 的合法值）
```

**验证方法**：

```python
meta = SLOT_REGISTRY[SlotName.HAND_OCCUPANCY]
print(meta.categories)  # ('empty', 'target', 'other')
print(meta.validate_value("target"))   # True
print(meta.validate_value("Target"))   # False - 大小写敏感
```

---

## 调试技巧

### 使用 InferenceLogger 调试推理问题

`InferenceLogger`（位于 `certvla/inference/logging.py`）是推理阶段的核心调试工具：

```python
from certvla.inference.logging import InferenceLogger

# 创建 verbose 模式的 logger（会输出每步详细信息到 Python 日志）
logger = InferenceLogger(verbose=True, max_episodes=100)

# 开始记录一个 episode
logger.begin_episode(metadata={"task": "pick_cup", "trial": 1})

# ... 推理过程中，RepairController 会自动调用 logger.log_step() ...

# 结束 episode
logger.end_episode()

# 获取诊断信息
trace = logger.get_last_trace()

# 查看摘要
print(trace.summary())
# {'num_steps': 45, 'num_repairs': 3, 'total_attempts': 48,
#  'mean_gap': 0.12, 'max_gap': 0.45, 'num_warnings': 0}

# 查看每步 gap 变化
print(trace.gap_history)
# [0.05, 0.08, 0.12, 0.45, 0.10, ...]

# 查看所有 repair 尝试（包括被拒绝的）
for record in trace.all_attempts:
    if not record.accepted:
        print(f"  被拒绝的尝试 #{record.repair_attempt}: "
              f"gap={record.gap.aggregated.mean().item():.4f}")

# 查看告警信息
for warning in trace.warnings:
    print(f"WARNING: {warning}")
```

---

### 监控 gate_value 诊断状态更新

状态令牌的门控值 `gate` 反映了模型对新旧状态的混合策略：

```python
# 在 CertVLAWrapper 前向传播后
z_t, gate = state_token_module.gated_update(tilde_z_t, z_prev)

gate_mean = gate.mean().item()
gate_std = gate.std().item()

print(f"Gate 均值: {gate_mean:.4f}, 标准差: {gate_std:.4f}")
```

**诊断指南**：

| gate_mean 范围 | 含义 | 可能的问题 |
|----------------|------|------------|
| 0.45 -- 0.55 | 正常（训练初期） | 无，门控处于中性状态 |
| 0.7 -- 1.0 | 偏向新状态 | 如果训练初期就出现，可能说明 z_prev 信息量不足 |
| 0.0 -- 0.3 | 偏向旧状态 | 模型可能忽略了当前观测，检查输入是否正确 |
| 恒定为 0.5 | 门控未学习 | 检查门控层的梯度是否为零（可能被冻结） |

---

### 拆分查看 Loss 各分量

当总 loss 表现异常时，逐分量检查是最有效的诊断手段：

```python
from certvla.training.losses import (
    cert_state_loss, cert_role_loss, cert_goal_loss,
    cert_action_loss, cert_consistency_loss,
    cert_dependence_loss, cert_counterfactual_loss,
    cert_total_loss,
)

# 逐项计算并打印
l_state = cert_state_loss(state_readout, state_target, mask, confidence)
l_role = cert_role_loss(role_logits, role_target, mask, confidence)
l_goal = cert_goal_loss(goal_preds, goal_target, role_target, mask, confidence)
l_act = cert_action_loss(pred_actions, expert_actions)
l_cons = cert_consistency_loss(state_readout, goal_preds, role_target, state_target_tH)
l_dep = cert_dependence_loss(expert_actions, actions_pos, actions_neg)
l_cf = cert_counterfactual_loss(z_t)  # v1: 通常返回 0

print(f"L_state: {l_state.item():.6f}")
print(f"L_role:  {l_role.item():.6f}")
print(f"L_goal:  {l_goal.item():.6f}")
print(f"L_act:   {l_act.item():.6f}")
print(f"L_cons:  {l_cons.item():.6f}")
print(f"L_dep:   {l_dep.item():.6f}")
print(f"L_cf:    {l_cf.item():.6f}")

# 使用 cert_total_loss 获取加权后的值
losses = {
    "state": l_state, "role": l_role, "goal": l_goal,
    "action": l_act, "consistency": l_cons,
    "dependence": l_dep, "counterfactual": l_cf,
}
weights = scheduler.get_loss_weights(current_step)
total, components = cert_total_loss(losses, weights)

print(f"\n加权后各分量:")
for name, val in components.items():
    print(f"  {name}: {val:.6f} (权重={weights.get(f'lambda_{name[0]}', 0):.2f})")
print(f"总 loss: {total.item():.6f}")
```

---

### 检查角色分类分布

如果角色分类不正常（例如所有槽位都预测为 `ignore`），可以检查 role logits 的分布：

```python
import torch.nn.functional as F
from certvla.slots.role_sets import J_CERT

for slot in J_CERT:
    if slot in role_logits:
        probs = F.softmax(role_logits[slot], dim=-1)  # (B, 3)
        mean_probs = probs.mean(dim=0)  # (3,)
        print(f"{slot.value}:")
        print(f"  advance={mean_probs[0]:.3f}, "
              f"preserve={mean_probs[1]:.3f}, "
              f"ignore={mean_probs[2]:.3f}")
```

**正常输出示例**（训练充分后）：

```
ee_target_proximity:
  advance=0.85, preserve=0.10, ignore=0.05
hand_occupancy:
  advance=0.02, preserve=0.90, ignore=0.08
target_contact:
  advance=0.70, preserve=0.20, ignore=0.10
```

---

### 快速定位 NaN 来源

如果训练中出现 NaN，使用以下代码逐步定位：

```python
# 在训练循环中启用异常检测
torch.autograd.set_detect_anomaly(True)

# 或者手动检查各输出
def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}!")
        print(f"  shape={tensor.shape}, "
              f"min={tensor[~torch.isnan(tensor)].min().item():.6f}, "
              f"max={tensor[~torch.isnan(tensor)].max().item():.6f}")
        return True
    return False

# 检查模型输出
check_nan(output.actions, "actions")
check_nan(z_t, "z_t")
for slot, readout in output.state_readout.items():
    check_nan(readout, f"state_readout[{slot.value}]")
```

**注意**：`torch.autograd.set_detect_anomaly(True)` 会显著降低训练速度，仅在调试时启用。

---

## 综合 FAQ

### Q: CertVLA 的 144 个单元测试覆盖了哪些内容？

测试分为 4 个 Phase，对应项目的 4 个开发阶段：

| 测试文件 | Phase | 覆盖内容 |
|----------|-------|----------|
| `test_slot_schema.py` | Phase 1 | 槽位定义、值域验证 |
| `test_slot_metrics.py` | Phase 1 | 槽位距离度量 |
| `test_certificate_mining.py` | Phase 1 | 证书挖掘算法 |
| `test_preserve_rules.py` | Phase 1 | 保持规则 |
| `test_goal_signature.py` | Phase 1 | 目标签名 |
| `test_model_shapes.py` | Phase 2 | 模型组件张量形状 |
| `test_losses.py` | Phase 3 | 7 个损失函数 |
| `test_inference.py` | Phase 4 | Gap 计算、Repair 控制器、Logger |

### Q: 如何在不安装完整依赖的情况下开发 CertVLA？

CertVLA 的最小依赖仅为 `torch`。以下操作不需要额外依赖：

```bash
# 最小安装
pip install torch

# 运行全部 CertVLA 测试
python -m pytest tests/ -v
```

不需要：TensorFlow、`prismatic`、`flash_attn`、LIBERO 仿真环境。

### Q: 如何复现论文中的基线结果？

1. 使用 `scripts/certvla/train_baseline.sh` 训练，保持默认超参数
2. 对每个任务集分别训练和评估
3. 使用 `SEED=42` 作为默认种子，额外使用 `SEED=0` 和 `SEED=1` 进行多种子实验
4. 报告 3 个种子的平均成功率和标准差

### Q: 训练时应该关注哪些指标？

按优先级排列：

1. **总 loss 下降趋势**：应该在前 10k 步明显下降
2. **L_state 收敛情况**：Stage 1 的基础，应在 5k 步内收敛
3. **L_role 的准确率**：Stage 2 引入后，角色分类准确率应快速上升
4. **L_act 下降幅度**：Stage 3 引入后，动作误差应持续下降
5. **gate_mean 变化**：应从约 0.5 逐渐偏移，说明门控在学习
6. **lambda_res 变化**：可学习残差缩放因子，反映粗细分支的贡献平衡

### Q: 评估一个模型需要多长时间？

以 LIBERO-Spatial (10 个任务 x 50 个 trial) 为例：

- 单 GPU (A100)：约 2--4 小时
- 每个 trial 约 30--60 秒（取决于任务复杂度）
- 启用 repair 后，推理时间增加约 10--30%（取决于 repair 触发频率）
