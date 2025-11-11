#!/usr/bin/env python3
"""
BBPE (Byte-Level BPE) 分词器使用示例

本示例展示了如何使用预训练的BBPE分词器进行文本编码和解码。
BBPE是BPE的一种变体，它在字节级别上进行操作，可以处理任意Unicode字符。
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
    print("BBPE分词器使用示例")
    print("=" * 50)
    
    # 创建BBPE分词器实例
    print("\n1. 创建BBPE分词器实例")
    tokenizer = zero_tokenizer.bbpe()
    print("✓ BBPE分词器创建成功")
    
    # 示例文本，包含各种Unicode字符
    text = "这是一个用于测试BBPE分词器的示例文本。Hello, world! 🌍"
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
    
    # BBPE的特点是能够处理任意Unicode字符
    print("\n8. BBPE特点测试:")
    special_chars = "😀🐍🌟αβγδε漢字𐍈"
    print(f"特殊字符: {special_chars}")
    special_tokens = tokenizer.encode(special_chars)
    print(f"编码结果: {special_tokens}")
    decoded_special = tokenizer.decode(special_tokens)
    print(f"解码结果: {decoded_special}")
    if special_chars == decoded_special:
        print("✓ 特殊字符编码解码一致")
    else:
        print("✗ 特殊字符编码解码不一致")
    
    print("\n示例完成!")


if __name__ == "__main__":
    main()