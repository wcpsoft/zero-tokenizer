#!/usr/bin/env python3
"""
Unigram 分词器训练示例

本示例展示了如何使用文本数据训练一个新的Unigram分词器。
Unigram语言模型是一种基于概率的分词方法，它通过EM算法学习每个子词的概率分布。
"""

import sys
import os
import tempfile
import random

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import zero_tokenizer
except ImportError:
    print("错误: 无法导入zero_tokenizer库。请确保已安装Python绑定。")
    print("可以尝试运行: pip install -e .")
    sys.exit(1)


def generate_sample_text():
    """生成示例训练文本"""
    chinese_sentences = [
        "自然语言处理是人工智能领域的一个重要方向。",
        "分词是自然语言处理的基础任务之一。",
        "Unigram算法是一种基于概率的分词方法。",
        "深度学习模型需要大量的文本数据进行训练。",
        "Transformer架构在NLP领域取得了巨大成功。",
        "预训练语言模型如BERT和GPT改变了NLP的研究范式。",
        "词嵌入技术能够将词语映射到高维向量空间。",
        "注意力机制是Transformer模型的核心组件。",
        "序列到序列模型适用于机器翻译等任务。",
        "语言模型能够预测下一个词的概率分布。"
    ]
    
    english_sentences = [
        "Natural language processing is an important field in artificial intelligence.",
        "Tokenization is one of the fundamental tasks in NLP.",
        "Unigram algorithm is a probabilistic tokenization method.",
        "Deep learning models require large amounts of text data for training.",
        "The Transformer architecture has achieved great success in the NLP field.",
        "Pre-trained language models like BERT and GPT have changed the NLP research paradigm.",
        "Word embedding techniques can map words to high-dimensional vector spaces.",
        "The attention mechanism is the core component of the Transformer model.",
        "Sequence-to-sequence models are suitable for tasks like machine translation.",
        "Language models can predict the probability distribution of the next word."
    ]
    
    # 混合中英文句子，并添加一些随机变化
    all_sentences = chinese_sentences + english_sentences
    random.shuffle(all_sentences)
    
    return all_sentences


def main():
    print("Unigram分词器训练示例")
    print("=" * 50)
    
    # 生成训练数据
    print("\n1. 生成训练数据...")
    sentences = generate_sample_text()
    print(f"✓ 生成了 {len(sentences)} 个训练句子")
    
    # 创建临时文件保存训练数据
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for sentence in sentences:
            f.write(sentence + '\n')
        training_file = f.name
    
    try:
        # 创建空的Unigram分词器
        print("\n2. 创建空的Unigram分词器...")
        tokenizer = zero_tokenizer.unigram()
        print("✓ Unigram分词器创建成功")
        
        # 设置训练参数
        vocab_size = 1000  # 词汇表大小
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]  # 特殊token
        
        print(f"\n3. 开始训练分词器...")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   特殊tokens: {special_tokens}")
        print(f"   训练文件: {training_file}")
        
        # 训练分词器
        tokenizer.train(
            files=[training_file],
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        
        print("✓ 分词器训练完成")
        
        # 测试训练后的分词器
        print("\n4. 测试训练后的分词器...")
        test_text = "这是一个测试Unigram分词器的句子。This is a test sentence."
        print(f"测试文本: {test_text}")
        
        # 编码
        tokens = tokenizer.encode(test_text)
        print(f"编码结果: {tokens}")
        
        # 解码
        decoded_text = tokenizer.decode(tokens)
        print(f"解码结果: {decoded_text}")
        
        # 验证一致性
        if test_text == decoded_text:
            print("✓ 编码解码一致")
        else:
            print("✗ 编码解码不一致")
        
        # 显示一些词汇表内容
        print("\n5. 词汇表示例 (前20个):")
        for i in range(min(20, tokenizer.get_vocab_size())):
            token = tokenizer.id_to_token(i)
            score = tokenizer.get_score(i) if hasattr(tokenizer, 'get_score') else None
            if score is not None:
                print(f"  ID {i}: '{token}' (概率: {score:.4f})")
            else:
                print(f"  ID {i}: '{token}'")
        
        # Unigram分词器的特点是可以获取token的概率分数
        print("\n6. Unigram分词器特点测试:")
        test_words = ["hello", "world", "测试", "分词", "language", "processing"]
        print("Token概率分数示例:")
        for word in test_words:
            word_tokens = tokenizer.encode(word)
            if word_tokens:
                token_id = word_tokens[0]
                token = tokenizer.id_to_token(token_id)
                score = tokenizer.get_score(token_id) if hasattr(tokenizer, 'get_score') else None
                if score is not None:
                    print(f"  '{token}': {score:.4f}")
                else:
                    print(f"  '{token}': 分数不可用")
        
        # 保存训练好的分词器
        model_file = "unigram_tokenizer.json"
        print(f"\n7. 保存训练好的分词器到 {model_file}...")
        tokenizer.save(model_file)
        print("✓ 分词器保存成功")
        
        # 加载保存的分词器
        print("\n8. 加载保存的分词器...")
        loaded_tokenizer = zero_tokenizer.unigram()
        loaded_tokenizer.load(model_file)
        print("✓ 分词器加载成功")
        
        # 验证加载的分词器
        loaded_tokens = loaded_tokenizer.encode(test_text)
        if tokens == loaded_tokens:
            print("✓ 加载的分词器与原始分词器一致")
        else:
            print("✗ 加载的分词器与原始分词器不一致")
        
    finally:
        # 清理临时文件
        os.unlink(training_file)
        if os.path.exists(model_file):
            os.unlink(model_file)
    
    print("\n示例完成!")


if __name__ == "__main__":
    main()