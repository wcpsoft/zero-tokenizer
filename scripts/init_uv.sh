#!/bin/bash

# 初始化uv环境的脚本
# 用于zero-tokenizer项目

set -e

echo "正在初始化uv环境..."

# 检查uv是否已安装
if ! command -v uv &> /dev/null; then
    echo "uv未安装，正在安装uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 检查uv是否在PATH中
if ! command -v uv &> /dev/null; then
    echo "错误: uv未在PATH中找到，请手动添加到PATH"
    exit 1
fi

# 创建虚拟环境
echo "创建Python虚拟环境..."
uv venv

# 激活虚拟环境并安装基本依赖
echo "激活虚拟环境并安装基本依赖..."
source .venv/bin/activate

# 安装maturin用于构建Python扩展
uv pip install maturin

# 安装开发依赖
uv pip install pytest pytest-benchmark

echo "uv环境初始化完成！"
echo "使用 'source .venv/bin/activate' 激活虚拟环境"