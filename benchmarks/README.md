# Benchmarks - 性能基准测试

## 📊 评价方法

本基准测试通过以下维度评价tokenizer性能：

1. **训练速度** - 从语料训练到构建词汇表的时间
2. **字典初始化** - 从预定义字典初始化的时间（仅Unigram和WordPiece）
3. **单条编码** - 单个文本编码的平均时间
4. **批量编码** - 批量文本的吞吐量（条/秒）
5. **解码速度** - token序列解码为文本的平均时间

## 🚀 调用方法

### 测试单个算法

```bash
# BPE算法
python benchmarks/compare_with_hf.py --algorithm bpe

# BBPE算法（字节级BPE）
python benchmarks/compare_with_hf.py --algorithm bbpe

# Unigram算法
python benchmarks/compare_with_hf.py --algorithm unigram

# WordPiece算法
python benchmarks/compare_with_hf.py --algorithm wordpiece
```

### 测试所有算法

```bash
python benchmarks/compare_with_hf.py --algorithm all
```

### 自定义参数

```bash
python benchmarks/compare_with_hf.py \
    --algorithm bpe \
    --vocab-size 5000 \
    --iterations 10
```

**参数说明**：
- `--algorithm`: 算法类型 (bpe/bbpe/unigram/wordpiece/all)
- `--vocab-size`: 词汇表大小，默认1000
- `--iterations`: 训练迭代次数，默认5

## 📦 依赖安装

```bash
# 安装Zero Tokenizer
maturin develop

# 安装HuggingFace tokenizers（用于对比）
uv pip install tokenizers
```

## 📈 输出说明

- **终端输出**: 实时显示测试进度和结果
- **JSON文件**: 保存详细数据到 `benchmark_{algorithm}_results.json`

---

**最后更新**: 2025-11-12
