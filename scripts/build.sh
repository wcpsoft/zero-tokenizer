#!/bin/bash

# 编译Rust代码的脚本
# 用于zero-tokenizer项目

set -e

echo "正在编译Rust代码..."

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

# 使用maturin构建Python扩展（这会自动编译Rust代码）
echo "构建Python扩展..."
maturin develop

echo "编译完成！"