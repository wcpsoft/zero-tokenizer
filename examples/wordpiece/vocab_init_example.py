#!/usr/bin/env python3
"""
示例：使用dict目录下的词表初始化WordPiece分词器

这个示例展示了如何使用dict目录下的词表文件来初始化WordPiece分词器。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def main():
    print("WordPiece分词器初始化词表示例")
    print("=" * 50)
    
    try:
        import zero_tokenizer
        print("✓ zero_tokenizer模块导入成功")
    except ImportError as e:
        print(f"✗ zero_tokenizer模块导入失败: {e}")
        return 1
    
    # 创建WordPiece分词器
    try:
        tokenizer = zero_tokenizer.WordPieceTokenizer()
        print("✓ WordPiece分词器创建成功")
    except Exception as e:
        print(f"✗ WordPiece分词器创建失败: {e}")
        return 1
    
    # 加载化学常用符号表
    try:
        tokenizer.load_vocab_from_dict("化学常用符号表.txt")
        print("✓ 成功加载化学常用符号表")
    except Exception as e:
        print(f"✗ 加载化学常用符号表失败: {e}")
        return 1
    
    # 显示词汇表大小
    vocab_size = tokenizer.get_vocab_size()
    print(f"当前词汇表大小: {vocab_size}")
    
    # 测试编码化学元素
    test_texts = ["氢", "Li", "氢Li", "氢氧化锂"]
    
    print("\n编码测试:")
    for text in test_texts:
        try:
            tokens = tokenizer.encode(text)
            print(f"文本: '{text}' -> tokens: {tokens}")
        except Exception as e:
            print(f"编码 '{text}' 失败: {e}")
    
    # 测试解码
    print("\n解码测试:")
    for text in test_texts:
        try:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            print(f"tokens: {tokens} -> 文本: '{decoded}'")
            if decoded != text:
                print(f"  ⚠️  解码结果不匹配!")
        except Exception as e:
            print(f"解码失败: {e}")
    
    # 加载常用汉字字表
    try:
        tokenizer.load_vocab_from_dict("常用汉字字表.txt")
        print("\n✓ 成功加载常用汉字字表")
    except Exception as e:
        print(f"\n✗ 加载常用汉字字表失败: {e}")
        return 1
    
    # 显示更新后的词汇表大小
    new_vocab_size = tokenizer.get_vocab_size()
    print(f"更新后词汇表大小: {new_vocab_size} (+{new_vocab_size - vocab_size})")
    
    # 测试编码中文文本
    chinese_texts = ["你好", "世界", "你好世界", "人工智能"]
    
    print("\n中文编码测试:")
    for text in chinese_texts:
        try:
            tokens = tokenizer.encode(text)
            print(f"文本: '{text}' -> tokens: {tokens}")
        except Exception as e:
            print(f"编码 '{text}' 失败: {e}")
    
    # 测试解码中文文本
    print("\n中文解码测试:")
    for text in chinese_texts:
        try:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            print(f"tokens: {tokens} -> 文本: '{decoded}'")
            if decoded != text:
                print(f"  ⚠️  解码结果不匹配!")
        except Exception as e:
            print(f"解码失败: {e}")
    
    print("\n" + "=" * 50)
    print("示例完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())