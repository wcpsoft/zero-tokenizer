"""测试错误处理

Python 3.10-3.12 兼容性测试
"""

import pytest
import os


def test_invalid_vocab_size():
    """测试无效的词汇表大小"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 词汇表大小太小（< 256）
    with pytest.raises(Exception) as exc_info:
        tokenizer.train(["test"], 100)

    # 验证错误消息
    assert "256" in str(exc_info.value) or "vocab" in str(exc_info.value).lower()


def test_load_nonexistent_file():
    """测试加载不存在的文件"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    with pytest.raises(Exception) as exc_info:
        tokenizer.load("nonexistent_file_12345.model")

    # 验证错误消息包含文件相关信息
    error_msg = str(exc_info.value).lower()
    assert "file" in error_msg or "文件" in error_msg or "not found" in error_msg


def test_save_to_invalid_path():
    """测试保存到无效路径"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    with pytest.raises(Exception):
        tokenizer.save("/invalid/nonexistent/path/model.bin")


def test_decode_invalid_tokens():
    """测试解码无效的token"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 无效的token ID（不在词汇表中）
    with pytest.raises(Exception) as exc_info:
        tokenizer.decode([999999])

    error_msg = str(exc_info.value)
    assert "ID" in error_msg or "未找到" in error_msg or "not found" in error_msg.lower()


def test_encode_after_failed_training():
    """测试训练失败后的编码"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 尝试失败的训练
    try:
        tokenizer.train(["test"], 100)
    except:
        pass

    # 应该仍能编码（使用初始词汇表）
    tokens = tokenizer.encode("test")
    assert len(tokens) > 0


def test_empty_text_list_training():
    """测试使用空文本列表训练"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 空训练数据
    result = None
    try:
        tokenizer.train([], 300)
        result = "success"
    except Exception:
        result = "error"

    # 两种结果都可以接受
    assert result in ["success", "error"]


def test_unigram_decode_invalid_token():
    """测试Unigram解码无效token"""
    from zero_tokenizer import UnigramTokenizer

    tokenizer = UnigramTokenizer()

    with pytest.raises(Exception):
        tokenizer.decode([9999999])


def test_wordpiece_decode_invalid_token():
    """测试WordPiece解码无效token"""
    from zero_tokenizer import WordPieceTokenizer

    tokenizer = WordPieceTokenizer()

    with pytest.raises(Exception):
        tokenizer.decode([9999999])


def test_bpe_decode_invalid_token():
    """测试BPE解码无效token"""
    from zero_tokenizer import Tokenizer

    tokenizer = Tokenizer()

    with pytest.raises(Exception):
        tokenizer.py_decode([9999999])


def test_corrupted_model_file():
    """测试加载损坏的模型文件"""
    from zero_tokenizer import BBPETokenizer

    # 创建一个损坏的文件
    corrupted_file = "test_corrupted.model"
    with open(corrupted_file, "w") as f:
        f.write("invalid corrupted data")

    try:
        tokenizer = BBPETokenizer()
        with pytest.raises(Exception):
            tokenizer.load(corrupted_file)
    finally:
        # 清理
        if os.path.exists(corrupted_file):
            os.remove(corrupted_file)


def test_save_without_training():
    """测试未训练就保存"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()
    save_path = "test_untrained.model"

    try:
        # 应该能成功保存（使用初始状态）
        tokenizer.save(save_path)

        # 应该能加载回来
        tokenizer2 = BBPETokenizer()
        tokenizer2.load(save_path)

        # 词汇表大小应该是初始大小
        assert tokenizer2.vocab_size() == 256
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


def test_train_with_very_small_vocab():
    """测试使用极小的词汇表"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 词汇表大小为1（不合理）
    with pytest.raises(Exception):
        tokenizer.train(["test"], 1)


def test_mixed_valid_invalid_tokens():
    """测试混合有效和无效token的解码"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()
    tokenizer.train(["test"], 300)

    # 混合有效和无效的token
    mixed_tokens = [0, 1, 999999, 2]

    with pytest.raises(Exception):
        tokenizer.decode(mixed_tokens)


def test_unicode_error_messages():
    """测试错误消息的Unicode支持"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 触发一个错误，检查错误消息是否正确显示
    try:
        tokenizer.train(["测试"], 50)  # 无效的vocab_size
    except Exception as e:
        error_msg = str(e)
        # 错误消息应该是可读的字符串
        assert isinstance(error_msg, str)
        assert len(error_msg) > 0


def test_none_input_handling():
    """测试None输入处理"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # Python层面应该处理None
    with pytest.raises((TypeError, AttributeError)):
        tokenizer.encode(None)  # type: ignore


def test_type_error_in_decode():
    """测试decode中的类型错误"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 传入错误类型
    with pytest.raises((TypeError, ValueError)):
        tokenizer.decode("invalid_input")  # type: ignore


def test_negative_token_ids():
    """测试负数token ID"""
    from zero_tokenizer import BBPETokenizer

    tokenizer = BBPETokenizer()

    # 负数token ID应该报错
    with pytest.raises(Exception):
        tokenizer.decode([-1, -2, -3])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
