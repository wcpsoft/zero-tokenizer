#!/usr/bin/env python3
"""
测试从dict目录加载初始化词表的功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_load_vocab_from_dict():
    """测试从dict目录加载初始化词表"""
    print("测试从dict目录加载初始化词表...")
    
    try:
        import zero_tokenizer
        print("✓ zero_tokenizer模块导入成功")
    except ImportError as e:
        print(f"✗ zero_tokenizer模块导入失败: {e}")
        return False
    
    # 创建BPE分词器
    try:
        tokenizer = zero_tokenizer.Tokenizer()
        print("✓ BPE分词器创建成功")
    except Exception as e:
        print(f"✗ BPE分词器创建失败: {e}")
        return False
    
    # 获取初始词汇表大小
    initial_vocab_size = tokenizer.get_vocab_size()
    print(f"初始词汇表大小: {initial_vocab_size}")
    
    # 测试加载化学基本元素表
    try:
        tokenizer.load_vocab_from_dict("化学基本元素表.txt")
        print("✓ 成功加载化学基本元素表")
    except Exception as e:
        print(f"✗ 加载化学基本元素表失败: {e}")
        return False
    
    # 检查词汇表是否增加
    after_elements_vocab_size = tokenizer.get_vocab_size()
    print(f"加载化学元素后词汇表大小: {after_elements_vocab_size}")
    
    if after_elements_vocab_size <= initial_vocab_size:
        print("✗ 词汇表大小没有增加")
        return False
    
    # 测试编码化学元素
    test_cases = [
        ("H", True),  # 应该能编码
        ("氢", True),  # 应该能编码
        ("Li", True),  # 应该能编码
        ("锂", True),  # 应该能编码
        ("氢Li", True),  # 应该能编码
        ("未知元素", False),  # 可能无法编码
    ]
    
    print("\n编码测试:")
    for text, should_succeed in test_cases:
        try:
            tokens = tokenizer.encode(text)
            if should_succeed:
                print(f"✓ 文本 '{text}' 编码成功: {tokens}")
            else:
                print(f"? 文本 '{text}' 编码成功（可能失败）: {tokens}")
        except Exception as e:
            if should_succeed:
                print(f"✗ 文本 '{text}' 编码失败: {e}")
                return False
            else:
                print(f"? 文本 '{text}' 编码失败（预期）: {e}")
    
    # 测试加载常用汉字字表
    try:
        tokenizer.load_vocab_from_dict("常用汉字字表.txt")
        print("\n✓ 成功加载常用汉字字表")
    except Exception as e:
        print(f"\n✗ 加载常用汉字字表失败: {e}")
        return False
    
    # 检查词汇表是否进一步增加
    final_vocab_size = tokenizer.get_vocab_size()
    print(f"加载常用汉字后词汇表大小: {final_vocab_size}")
    
    if final_vocab_size <= after_elements_vocab_size:
        print("✗ 词汇表大小没有增加")
        return False
    
    # 测试编码中文文本
    chinese_test_cases = [
        ("你", True),  # 应该能编码
        ("好", True),  # 应该能编码
        ("你好", True),  # 应该能编码
        ("世界", True),  # 应该能编码
        ("你好世界", True),  # 应该能编码
    ]
    
    print("\n中文编码测试:")
    for text, should_succeed in chinese_test_cases:
        try:
            tokens = tokenizer.encode(text)
            if should_succeed:
                print(f"✓ 文本 '{text}' 编码成功: {tokens}")
            else:
                print(f"? 文本 '{text}' 编码成功（可能失败）: {tokens}")
        except Exception as e:
            if should_succeed:
                print(f"✗ 文本 '{text}' 编码失败: {e}")
                return False
            else:
                print(f"? 文本 '{text}' 编码失败（预期）: {e}")
    
    # 测试编码解码往返
    print("\n编码解码往返测试:")
    test_texts = ["氢", "Li", "氢Li", "你", "好", "你好"]
    for text in test_texts:
        try:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            if decoded == text:
                print(f"✓ '{text}' -> {tokens} -> '{decoded}' (往返成功)")
            else:
                print(f"✗ '{text}' -> {tokens} -> '{decoded}' (往返失败)")
                return False
        except Exception as e:
            print(f"✗ '{text}' 往返测试失败: {e}")
            return False
    
    print("\n✓ 所有测试通过!")
    return True

def main():
    """运行所有测试"""
    print("开始测试初始化词表功能...")
    print("=" * 50)
    
    if test_load_vocab_from_dict():
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！")
        return 0
    else:
        print("\n" + "=" * 50)
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())