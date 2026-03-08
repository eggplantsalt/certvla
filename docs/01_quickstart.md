# CertVLA 快速开始指南

> 本文档带你从零开始，用最短路径验证 CertVLA 是否安装正确。

## 前置条件

- Python 3.10+（推荐 3.10.14）
- PyTorch 2.2.0
- CUDA 11.8+（训练需要，纯测试不需要 GPU）
- conda 或 virtualenv

## 第一步：创建环境

```bash
# 创建 conda 环境
conda create -n certvla python=3.10.14 -y
conda activate certvla
```

## 第二步：安装依赖

```bash
# 进入项目目录
cd /path/to/openvla-oft

# 安装项目（editable 模式）
pip install -e .

# 如果上面安装 tensorflow 报错，可以先跳过，CertVLA 核心测试不需要 tensorflow
# tensorflow 只在加载 RLDS 数据集时需要
```

## 第三步：验证安装

```bash
# 运行冒烟测试（不需要 GPU，不需要数据集）
bash scripts/certvla/smoke_test.sh

# 或者直接运行 pytest
python -m pytest tests/ -v --tb=short
```

**预期结果**：144 个测试全部通过，输出类似：

```
======================= 144 passed in 1.56s ========================
```

## 第四步：运行特定模块测试

```bash
# 只测试 slot 体系（Phase 1）
bash scripts/certvla/run_tests.sh data

# 只测试模型层（Phase 2）
bash scripts/certvla/run_tests.sh model

# 只测试损失函数（Phase 3）
bash scripts/certvla/run_tests.sh losses

# 只测试推理逻辑（Phase 4）
bash scripts/certvla/run_tests.sh inference
```

## 第五步：查看核心代码

推荐从以下文件开始阅读：

```bash
# 1. slot 词汇表定义（10 个 task-relative slot）
# 这是整个系统的基础数据结构
cat certvla/slots/schema.py

# 2. CertVLA wrapper 的前向流程
# 这是理解整个模型架构的关键入口
cat certvla/model/certvla_wrapper.py

# 3. 7 个损失函数
cat certvla/training/losses.py
```

## 常见安装问题

### `ModuleNotFoundError: No module named 'draccus'`

这是 `prismatic` 的依赖。运行 `pip install -e .` 会自动安装。如果只想运行 CertVLA 测试，`certvla/` 模块已经隔离了对 prismatic 的依赖，不需要 draccus。

### `ModuleNotFoundError: No module named 'tensorflow'`

tensorflow 只在加载 RLDS 数据集时需要。CertVLA 的单元测试不依赖 tensorflow。如果只是验证安装，可以暂时跳过。

### 测试全部通过但有 UserWarning

如果看到 `UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor...`，这是正常的 PyTorch 警告，不影响测试结果。

## 下一步

- 了解项目理论：阅读 `docs/02_theory_and_core_idea.md`
- 了解代码结构：阅读 `docs/03_code_structure.md`
- 准备训练：阅读 `docs/06_training_pipeline.md`
