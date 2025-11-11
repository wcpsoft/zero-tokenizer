# Zero Tokenizer

一个高性能的LLM模型分词器库，使用Rust语言实现，支持多种分词算法，包括BPE、BBPE、Unigram和WordPiece。

## 特性

- 🚀 **高性能**：使用Rust实现，提供极致的性能
- 📚 **多算法支持**：支持BPE、BBPE、Unigram和WordPiece四种主流分词算法
- 🌐 **多语言支持**：支持多种语言的文本处理，包括中文、英文等
- 🔧 **灵活配置**：支持自定义词汇表大小、特殊token和正则表达式模式
- 🐍 **Python绑定**：提供Python接口，方便在Python项目中使用
- 📊 **并行处理**：使用Rayon库实现并行处理，提高训练效率
- 🔄 **流式训练**：支持从迭代器流式训练，适用于大规模数据集

## 安装

### Rust

在您的`Cargo.toml`中添加：

```toml
[dependencies]
zero-tokenizer = "0.1.0"
```

### Python

```bash
pip install zero-tokenizer
```

## 快速开始

### BPE分词器

```python
from zero_tokenizer import BPETokenizer

# 创建分词器
tokenizer = BPETokenizer()

# 训练分词器
tokenizer.train(
    files=["path/to/your/data.txt"],
    vocab_size=30000,
    special_tokens=["<unk>", "<s>", "</s>"]
)

# 编码文本
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# 解码tokens
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")
```

### BBPE分词器

```python
from zero_tokenizer import BBPETokenizer

# 创建分词器
tokenizer = BBPETokenizer()

# 训练分词器
tokenizer.train(
    files=["path/to/your/data.txt"],
    vocab_size=50000,
    special_tokens=["<unk>", "<s>", "</s>"]
)

# 编码文本
text = "你好，世界！"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# 解码tokens
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")
```

### Unigram分词器

```python
from zero_tokenizer import UnigramTokenizer

# 创建分词器
tokenizer = UnigramTokenizer()

# 训练分词器
tokenizer.train(
    files=["path/to/your/data.txt"],
    vocab_size=30000,
    special_tokens=["<unk>", "<s>", "</s>"]
)

# 编码文本
text = "这是一个测试文本"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# 解码tokens
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")
```

### WordPiece分词器

```python
from zero_tokenizer import WordPieceTokenizer

# 创建分词器
tokenizer = WordPieceTokenizer()

# 训练分词器
tokenizer.train(
    files=["path/to/your/data.txt"],
    vocab_size=30000,
    special_tokens=["<unk>", "<s>", "</s>"]
)

# 编码文本
text = "这是一个测试文本"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# 解码tokens
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")
```

## 算法介绍

### BPE (Byte Pair Encoding)

BPE是一种基于频率的子词分词算法，通过迭代合并最频繁出现的字符对来构建词汇表。它简单高效，易于控制词汇表大小，被广泛应用于GPT系列模型中。

[详细文档](docs/bpe.md)

### BBPE (Byte-Level BPE)

BBPE是BPE的字节级变体，直接在字节级别进行操作，能够处理任意Unicode字符，包括表情符号和特殊字符。

[详细文档](docs/bbpe.md)

### Unigram

Unigram是一种基于概率的语言模型，从一个大的初始词汇表开始，逐步移除不重要的子词，直到达到目标词汇表大小。它支持多种分词策略，灵活性高。

[详细文档](docs/unigram.md)

### WordPiece

WordPiece是一种基于概率的子词分词算法，选择能够最大化语言模型似然的子词合并。它被广泛应用于BERT等预训练语言模型中。

[详细文档](docs/wordpiece.md)

## 示例

在`examples`目录中提供了每种算法的详细示例，包括：

- `examples/bpe/`: BPE算法示例
  - `usage_example.py`: BPE分词器使用示例
  - `training_example.py`: BPE分词器训练示例
- `examples/bbpe/`: BBPE算法示例
  - `usage_example.py`: BBPE分词器使用示例
  - `training_example.py`: BBPE分词器训练示例
- `examples/unigram/`: Unigram算法示例
  - `usage_example.py`: Unigram分词器使用示例
  - `training_example.py`: Unigram分词器训练示例
- `examples/wordpiece/`: WordPiece算法示例
  - `usage_example.py`: WordPiece分词器使用示例
  - `training_example.py`: WordPiece分词器训练示例

## 性能

| 算法 | 训练速度 | 推理速度 | 词汇表大小 | 支持语言 |
|------|----------|----------|------------|----------|
| BPE | 快 | 快 | 可控 | 多语言 |
| BBPE | 中等 | 快 | 较大 | 多语言 |
| Unigram | 慢 | 中等 | 可控 | 多语言 |
| WordPiece | 慢 | 快 | 可控 | 多语言 |

## 开发

### 构建项目

```bash
# 构建Rust库
cargo build --release

# 构建Python绑定
maturin develop
```

### 运行测试

```bash
# 运行Rust测试
cargo test

# 运行Python测试
python -m pytest tests/
```

## 贡献

欢迎贡献代码！请遵循以下步骤：
```bash
1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/xxx`)
3. 提交您的更改 (`git commit -m 'Add some xxx feature'`)
4. 推送到分支 (`git push origin feature/xxx`)
5. 打开一个 Pull Request
```
## 许可证

本项目采用双重许可证 - 您可以选择使用 Apache 2.0 许可证或 MIT 许可证。详细信息请查看 [LICENSE](LICENSE.md) 文件。

## 致谢
感谢 [@code-your-own-llm](https://github.com/datawhalechina/code-your-own-llm)、[@nanochat](https://github.com/karpathy/nanochat) 提供灵感和参考, 以及感谢CME295课程的教材、PPT等资源,推荐购买电子书[Super Study Guide: Transformer 与大语言模型](https://leanpub.com/transformer-da-yuyan-moxing/)。


## 更新日志

### v0.1.0 (2025-11-11)

- 初始版本发布
- 支持BPE、BBPE、Unigram和WordPiece四种算法
- 提供Python绑定
- 添加详细文档和示例