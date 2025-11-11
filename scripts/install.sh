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

# 使用maturin安装包
echo "使用maturin安装Python包..."
maturin develop

echo "Python包安装完成！"