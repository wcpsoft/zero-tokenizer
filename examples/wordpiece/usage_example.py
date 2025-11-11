#!/usr/bin/env python3
"""
WordPiece 分词器使用示例

本示例展示了如何使用预训练的WordPiece分词器进行文本编码和解码。
WordPiece是一种基于子词的分词方法，它通过最大化训练数据的似然函数来学习词汇表。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import zero_tokenizer
except ImportError:
    print("错误: 无法导入zero_tokenizer库。请确保已安装Python绑定。")
    print("可以尝试运行: pip install -e .")
    sys.exit(1)


def main():
    print("WordPiece分词器使用示例")
    print("=" * 50)
    
    # 创建WordPiece分词器实例
    print("\n1. 创建WordPiece分词器实例")
    tokenizer = zero_tokenizer.wordpiece()
    print("✓ WordPiece分词器创建成功")
    
    # 示例文本
    text = "这是一个用于测试WordPiece分词器的示例文本。Hello, world! 🌍"
    print(f"\n2. 原始文本: {text}")
    
    # 编码文本
    print("\n3. 编码文本...")
    tokens = tokenizer.encode(text)
    print(f"✓ 编码成功，得到 {len(tokens)} 个token")
    print(f"Token IDs: {tokens}")
    
    # 解码文本
    print("\n4. 解码token...")
    decoded_text = tokenizer.decode(tokens)
    print(f"✓ 解码成功")
    print(f"解码文本: {decoded_text}")
    
    # 验证编码解码的一致性
    print("\n5. 验证编码解码一致性...")
    if text == decoded_text:
        print("✓ 编码解码一致，无损转换")
    else:
        print("✗ 编码解码不一致，可能存在信息损失")
        print(f"原始: {text}")
        print(f"解码: {decoded_text}")
    
    # 展示token到单词的映射
    print("\n6. Token到单词的映射:")
    for i, token_id in enumerate(tokens[:10]):  # 只显示前10个token
        token = tokenizer.id_to_token(token_id)
        print(f"  Token {i+1}: ID={token_id}, Token='{token}'")
    
    # 统计信息
    print("\n7. 分词器统计信息:")
    print(f"  词汇表大小: {tokenizer.get_vocab_size()}")
    print(f"  特殊token数量: {len(tokenizer.get_special_tokens())}")
    
    # WordPiece的特点是使用##表示子词的延续
    print("\n8. WordPiece特点测试:")
    test_words = ["unhappiness", "preprocessing", "tokenization", "unbelievable"]
    print("子词分割示例:")
    for word in test_words:
        word_tokens = tokenizer.encode(word)
        word_str_tokens = [tokenizer.id_to_token(t) for t in word_tokens]
        print(f"  '{word}': {' '.join(word_str_tokens)}")
    
    print("\n示例完成!")


if __name__ == "__main__":
    main()