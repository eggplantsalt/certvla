#!/bin/bash
# ============================================================================
# CertVLA Smoke Test 脚本
# ============================================================================
# 用途：快速验证 CertVLA 安装和核心逻辑是否正常
# 使用方法：
#   bash scripts/certvla/smoke_test.sh
# 预期结果：144 个测试全部通过
# 注意：此脚本不需要 GPU，不需要数据集，纯 CPU 即可运行
# ============================================================================

cd "$(dirname "$0")/../.."   # 切换到项目根目录

echo "========================================="
echo "  CertVLA Smoke Test"
echo "========================================="
echo ""
echo "1. 检查 Python 环境..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import certvla; print('  certvla 包: OK')"
echo ""

echo "2. 运行全部单元测试..."
python -m pytest tests/ -v --tb=short 2>&1

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "  ✓ Smoke Test 全部通过"
    echo "========================================="
else
    echo "========================================="
    echo "  ✗ 存在失败的测试，请检查上方输出"
    echo "========================================="
fi

exit $EXIT_CODE
