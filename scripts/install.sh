#!/bin/bash

# 安装Python包的脚本
# 用于zero-tokenizer项目

set -e

echo "正在安装Python包..."

# 检查是否在项目根目录
if [ ! -f "Cargo.toml" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "激活Python虚拟环境..."
    source .venv/bin/activate
fi

# 使用uv或pip安装依赖
if command -v uv &> /dev/null; then
    echo "使用uv安装依赖..."
    uv pip install -e .
else
    echo "使用pip安装依赖..."
    pip install -e .
fi

echo "Python包安装完成！"