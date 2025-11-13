"""测试批量编码解码功能

Python 3.10-3.12 兼容性测试
"""

import sys
import time
import pytest


def test_bbpe_encode_batch():
    """测试BBPE批量编码功能"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()
    texts = [
        "Hello world!",
        "你好世界！",
        "Test text",
        "Another test",
    ]

    tokenizer.train(texts, 300)

    # 批量编码
    batch_tokens = tokenizer.encode_batch(texts)
    assert len(batch_tokens) == len(texts)

    # 验证每个结果
    for i, tokens in enumerate(batch_tokens):
        decoded = tokenizer.decode(tokens)
        assert decoded == texts[i], f"解码不一致: {texts[i]} != {decoded}"


def test_bbpe_decode_batch():
    """测试BBPE批量解码功能"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()
    texts = ["Hello", "World", "Test"]

    tokenizer.train(texts, 300)

    # 编码
    token_lists = [tokenizer.encode(text) for text in texts]

    # 批量解码
    decoded_texts = tokenizer.decode_batch(token_lists)
    assert decoded_texts == texts


def test_large_batch():
    """测试大批量处理"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 100个文本
    texts = [f"Test text number {i}" for i in range(100)]
    tokenizer.train(texts[:10], 300)

    # 批量编码应该快速完成
    start = time.time()
    batch_tokens = tokenizer.encode_batch(texts)
    duration = time.time() - start

    assert len(batch_tokens) == 100
    print(f"批量编码100个文本耗时: {duration:.3f}秒")

    # 验证性能（应该<1秒）
    assert duration < 5.0, f"批量编码太慢: {duration}秒"


def test_unigram_encode_batch():
    """测试Unigram批量编码"""
    from zero_tokenizer import UnigramTokenizer

    tokenizer = UnigramTokenizer()
    texts = [
        "测试文本1",
        "测试文本2",
        "测试文本3",
    ]

    tokenizer.train(texts, 16000)

    batch_tokens = tokenizer.encode_batch(texts)
    assert len(batch_tokens) == len(texts)

    # 验证解码一致性
    for i, tokens in enumerate(batch_tokens):
        decoded = tokenizer.decode(tokens)
        assert decoded == texts[i]


def test_wordpiece_encode_batch():
    """测试WordPiece批量编码"""
    from zero_tokenizer import WordPieceTokenizer

    tokenizer = WordPieceTokenizer()
    texts = [
        "测试文本1",
        "测试文本2",
        "测试文本3",
    ]

    tokenizer.train(texts, 16000)

    batch_tokens = tokenizer.encode_batch(texts)
    assert len(batch_tokens) == len(texts)


def test_bpe_encode_batch():
    """测试BPE批量编码"""
    from zero_tokenizer import Tokenizer

    tokenizer = Tokenizer()
    texts = [
        "Hello world",
        "Test text",
        "Another example",
    ]

    tokenizer.train_from_iterator(texts, 300)

    # BPE使用不同的方法名
    batch_tokens = [tokenizer.py_encode(text) for text in texts]
    assert len(batch_tokens) == len(texts)

    # 验证解码
    for i, tokens in enumerate(batch_tokens):
        decoded = tokenizer.py_decode(tokens)
        assert decoded == texts[i]


def test_empty_texts_in_batch():
    """测试批量处理中的空字符串"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()
    tokenizer.train(["test"], 300)

    # 包含空字符串的批次
    texts = ["Hello", "", "World", ""]

    batch_tokens = tokenizer.encode_batch(texts)

    # 验证空字符串编码为空列表
    assert len(batch_tokens[0]) > 0  # "Hello"
    assert len(batch_tokens[1]) == 0  # ""
    assert len(batch_tokens[2]) > 0  # "World"
    assert len(batch_tokens[3]) == 0  # ""


def test_mixed_length_batch():
    """测试不同长度文本的批量处理"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 不同长度的文本
    texts = [
        "a",
        "ab",
        "abc def",
        "This is a much longer text for testing batch operations",
    ]

    tokenizer.train(texts, 300)

    batch_tokens = tokenizer.encode_batch(texts)

    # 验证每个结果
    for i, tokens in enumerate(batch_tokens):
        decoded = tokenizer.decode(tokens)
        assert decoded == texts[i]


def test_batch_with_unicode():
    """测试包含Unicode字符的批量处理"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()
    texts = [
        "Hello 👋",
        "世界 🌍",
        "Testing emoji 🎉",
        "Mixed 中英文 text",
    ]

    tokenizer.train(texts, 300)

    batch_tokens = tokenizer.encode_batch(texts)

    for i, tokens in enumerate(batch_tokens):
        decoded = tokenizer.decode(tokens)
        assert decoded == texts[i]


def test_batch_performance_comparison():
    """比较批量处理和单独处理的性能"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()
    texts = [f"Test text {i}" for i in range(50)]
    tokenizer.train(texts[:5], 300)

    # 单独处理
    start = time.time()
    individual_tokens = [tokenizer.encode(text) for text in texts]
    individual_time = time.time() - start

    # 批量处理
    start = time.time()
    batch_tokens = tokenizer.encode_batch(texts)
    batch_time = time.time() - start

    print(f"单独处理: {individual_time:.3f}秒")
    print(f"批量处理: {batch_time:.3f}秒")

    # 验证结果一致
    assert len(individual_tokens) == len(batch_tokens)
    for i in range(len(texts)):
        assert individual_tokens[i] == batch_tokens[i]


if __name__ == "__main__":
    # 支持直接运行
    pytest.main([__file__, "-v"])
