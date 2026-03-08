# =============================================================================
# CertVLA 逐 Slot 距离度量与值-张量转换工具 (metrics.py)
# =============================================================================
#
# 【模块在整体流水线中的位置】
#   CertVLA 需要对比"预测的 slot 状态"和"真实的 slot 状态"之间的差异，
#   以判定 certificate 是否满足（即当前 action chunk 是否导致了期望的后果）。
#   本模块提供两大能力：
#     1. 距离度量 d_j(a, b) —— 量化单个 slot 的两个值之间的差异（见论文 8.2 节）
#     2. 值-张量互转 —— 将人类可读的 slot 值（int/float/str）转为神经网络可处理的
#        numpy 向量，或反向还原。
#
# 【张量形状约定】
#   • 单个 slot 的张量：
#       - BINARY / CONTINUOUS / CONFIDENCE → shape = (1,), dtype = float32
#       - CATEGORICAL → shape = (num_categories,), dtype = float32, one-hot 编码
#   • 完整 slot 状态的 flat tensor (slot_state_to_flat_tensor):
#       按 SlotName 枚举顺序拼接所有 slot 的张量，总维度 = flat_tensor_dim()。
#       v1 的 10 个 slot 计算如下：
#           1(ee_target_proximity) + 3(hand_occupancy) + 1(target_contact)
#         + 1(target_goal_proximity) + 3(support_relation) + 3(containment_relation)
#         + 1(articulation_progress) + 1(orientation_alignment) + 1(completion_latch)
#         + 1(task_visible_confidence) = 16 维
#
# 【距离度量设计决策】
#   - 二值型和连续型统一使用 L1 距离 |a - b|，结果总在 [0, 1] 内。
#   - 分类型使用 Hamming 距离 (相同=0, 不同=1)，因为类别间没有有序关系。
#   - 所有距离都归一化到 [0, 1]，这使得不同 slot 之间可以直接做加权和。
#
# 【常见陷阱】
#   1. 缺失 slot 在 flat tensor 中用零向量表示——这对 binary slot 而言等价于 "0"
#      （即假），对 categorical slot 而言是全零（不代表任何类别），需注意下游解读。
#   2. tensor_to_slot_value 对 categorical slot 做 argmax，如果输入是全零向量，
#      argmax 返回 index=0 (即第一个类别)，这是一种默认行为但未必正确。
#   3. slot_state_to_flat_tensor 的拼接顺序由 SlotName 枚举顺序决定，
#      如果 schema.py 中枚举顺序改变，已持久化的张量就不兼容了。
# =============================================================================

"""
Per-slot distance functions and value conversion utilities.

Distance functions d_j(a, b) per context doc section 8.2:
- Binary: |a - b| (0 or 1)
- Categorical: 1 - (a == b) (Hamming)
- Continuous: |a - b| (L1 in [0,1])
"""

from typing import Union

import numpy as np

from certvla.slots.schema import SlotDomain, SlotMeta, SlotName, SLOT_REGISTRY, get_slot_meta


# ---------------------------------------------------------------------------
# 公开接口: slot_distance —— 计算单个 slot 的距离 d_j(a, b)
# ---------------------------------------------------------------------------
def slot_distance(slot: SlotName, a: Union[int, float, str], b: Union[int, float, str]) -> float:
    """Compute per-slot distance d_j(a, b).

    对应论文 8.2 节的距离函数定义。返回值始终在 [0, 1] 内。
    用途：certificate 判定时检查 slot 的实际变化是否符合预期。

    Args:
        slot: The slot name.
        a: First value. （通常是 predicted / current 值）
        b: Second value.（通常是 expected / target 值）

    Returns:
        Distance in [0, 1].
    """
    meta = get_slot_meta(slot)
    return _distance_by_domain(meta, a, b)


# ---------------------------------------------------------------------------
# 内部实现: 按值域类型分派距离计算
# ---------------------------------------------------------------------------
def _distance_by_domain(meta: SlotMeta, a: Union[int, float, str], b: Union[int, float, str]) -> float:
    """根据 slot 的值域类型选择合适的距离度量。

    - BINARY:     |a - b|, 结果为 0.0 或 1.0
    - CATEGORICAL: Hamming 距离, 相同=0, 不同=1
    - CONTINUOUS / CONFIDENCE: L1 距离 |a - b|, 已在 [0,1] 内
    """
    if meta.domain == SlotDomain.BINARY:
        return abs(float(a) - float(b))
    elif meta.domain == SlotDomain.CATEGORICAL:
        return 0.0 if a == b else 1.0
    elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        return abs(float(a) - float(b))
    else:
        raise ValueError(f"Unknown domain {meta.domain}")


# ---------------------------------------------------------------------------
# slot 值 → 张量: 将人类可读的 slot 值编码为 numpy 向量
# ---------------------------------------------------------------------------
def slot_value_to_tensor(slot: SlotName, value: Union[int, float, str]) -> np.ndarray:
    """Convert a slot value to its tensor representation.

    - Binary: [float] (0.0 or 1.0)          → shape (1,)
    - Categorical: one-hot vector of length num_categories → shape (num_categories,)
    - Continuous/Confidence: [float]         → shape (1,)

    【注意】分类型的 one-hot 编码中，标签到索引的映射由 meta.categories 的声明顺序决定，
    例如 hand_occupancy 的 ("empty", "target", "other") 对应索引 (0, 1, 2)。
    """
    meta = get_slot_meta(slot)

    if meta.domain == SlotDomain.BINARY:
        return np.array([float(value)], dtype=np.float32)  # shape: (1,)
    elif meta.domain == SlotDomain.CATEGORICAL:
        idx = meta.categories.index(value)  # 如果 value 不在 categories 中会抛 ValueError
        one_hot = np.zeros(len(meta.categories), dtype=np.float32)  # shape: (num_categories,)
        one_hot[idx] = 1.0
        return one_hot
    elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        return np.array([float(value)], dtype=np.float32)  # shape: (1,)
    else:
        raise ValueError(f"Unknown domain {meta.domain}")


# ---------------------------------------------------------------------------
# 张量 → slot 值: 将 numpy 向量解码为人类可读的 slot 值
# ---------------------------------------------------------------------------
def tensor_to_slot_value(slot: SlotName, tensor: np.ndarray) -> Union[int, float, str]:
    """Convert tensor representation back to slot value.

    - Binary: round to 0 or 1          (四舍五入取整)
    - Categorical: argmax -> category string  (取最大分量对应的标签)
    - Continuous/Confidence: clamp to valid_range  (截断到 [0, 1])

    【陷阱】如果 categorical slot 的 tensor 是全零向量，argmax 返回 0，
    即默认选择第一个类别——这未必语义正确，调用方需自行处理缺失值。
    """
    meta = get_slot_meta(slot)

    if meta.domain == SlotDomain.BINARY:
        return int(round(float(tensor[0])))  # 0 或 1
    elif meta.domain == SlotDomain.CATEGORICAL:
        idx = int(np.argmax(tensor))  # 找到 one-hot 中值最大的维度
        return meta.categories[idx]   # 返回对应的字符串标签
    elif meta.domain in (SlotDomain.CONTINUOUS, SlotDomain.CONFIDENCE):
        v = float(tensor[0])
        lo, hi = meta.valid_range     # 默认 (0.0, 1.0)
        return max(lo, min(hi, v))    # clamp 截断
    else:
        raise ValueError(f"Unknown domain {meta.domain}")


# ---------------------------------------------------------------------------
# 完整 slot 状态 → 扁平化张量
# ---------------------------------------------------------------------------
def slot_state_to_flat_tensor(values: dict) -> np.ndarray:
    """Convert a dict of {SlotName: value} to a flat numpy array.

    Ordering follows SlotName enum order. Binary/continuous slots contribute 1 dim;
    categorical slots contribute num_categories dims (one-hot).

    将所有 slot 的张量按 SlotName 枚举顺序拼接为一个一维向量。
    这个 flat tensor 可以直接作为神经网络的输入/目标。

    【张量布局示例 (v1, 共 16 维)】
        [ee_target_proximity(1)] [hand_occupancy(3)] [target_contact(1)]
        [target_goal_proximity(1)] [support_relation(3)] [containment_relation(3)]
        [articulation_progress(1)] [orientation_alignment(1)] [completion_latch(1)]
        [task_visible_confidence(1)]

    【缺失 slot 处理】
        如果 values 字典中缺少某个 slot，该位置用零向量填充。
        这意味着 categorical slot 的缺失值是全零（非合法 one-hot），
        binary slot 的缺失值等价于 0（"否" / "关"），需注意语义。
    """
    parts = []
    for slot_name in SlotName:
        if slot_name in values:
            parts.append(slot_value_to_tensor(slot_name, values[slot_name]))
        else:
            meta = get_slot_meta(slot_name)
            # 缺失 slot: 用合适维度的零向量代替
            if meta.domain == SlotDomain.CATEGORICAL:
                parts.append(np.zeros(len(meta.categories), dtype=np.float32))
            else:
                parts.append(np.zeros(1, dtype=np.float32))
    return np.concatenate(parts)  # shape: (flat_tensor_dim(),)


# ---------------------------------------------------------------------------
# 计算 flat tensor 的总维度
# ---------------------------------------------------------------------------
def flat_tensor_dim() -> int:
    """Return the total dimensionality of the flat slot state tensor.

    遍历所有 slot，累加每个 slot 贡献的维度数。
    v1 的 10 个 slot 总维度 = 16。
    此函数用于初始化神经网络层的输入/输出维度。
    """
    dim = 0
    for slot_name in SlotName:
        meta = get_slot_meta(slot_name)
        if meta.domain == SlotDomain.CATEGORICAL:
            dim += len(meta.categories)  # one-hot 编码的维度
        else:
            dim += 1                     # 标量: 1 维
    return dim
