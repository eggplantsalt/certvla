#!/bin/bash
# ============================================================================
# CertVLA 单元测试运行脚本
# ============================================================================
# 用途：运行所有 CertVLA 单元测试（Phase 1-4）
# 使用方法：
#   bash scripts/certvla/run_tests.sh           # 运行全部测试
#   bash scripts/certvla/run_tests.sh losses     # 只运行 loss 相关测试
#   bash scripts/certvla/run_tests.sh inference   # 只运行推理相关测试
# ============================================================================

cd "$(dirname "$0")/../.."   # 切换到项目根目录

if [ -z "$1" ]; then
    echo "========================================="
    echo "  运行全部 CertVLA 测试"
    echo "========================================="
    python -m pytest tests/ -v --tb=short
elif [ "$1" = "losses" ]; then
    echo "运行 Phase 3 损失函数测试..."
    python -m pytest tests/test_losses.py -v --tb=short
elif [ "$1" = "inference" ]; then
    echo "运行 Phase 4 推理测试..."
    python -m pytest tests/test_inference.py -v --tb=short
elif [ "$1" = "model" ]; then
    echo "运行 Phase 2 模型测试..."
    python -m pytest tests/test_model_shapes.py -v --tb=short
elif [ "$1" = "data" ]; then
    echo "运行 Phase 1 数据层测试..."
    python -m pytest tests/test_slot_schema.py tests/test_slot_metrics.py \
        tests/test_certificate_mining.py tests/test_preserve_rules.py \
        tests/test_goal_signature.py -v --tb=short
else
    echo "运行匹配 '$1' 的测试..."
    python -m pytest tests/ -v --tb=short -k "$1"
fi
