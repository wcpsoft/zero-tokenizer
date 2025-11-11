"""
测试分词器的训练功能
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_bpe_training():
    """测试BPE分词器的训练功能"""
    try:
        from zero_tokenizer import Tokenizer
        
        # 创建分词器实例
        tokenizer = Tokenizer()
        
        # 准备训练数据
        training_texts = [
            "Hello world!",
            "Hello there!",
            "How are you?",
            "I am fine.",
            "Thank you!",
            "你好世界！",
            "你好吗？",
            "我很好，谢谢！",
            "不客气！",
            "再见！"
        ]
        
        # 训练分词器
        tokenizer.train_from_iterator(training_texts, 300)
        
        # 测试编码
        test_text = "Hello, 你好!"
        tokens = tokenizer.py_encode(test_text)
        print(f"✓ BPE训练后编码成功: {test_text} -> {tokens}")
        
        # 测试解码
        decoded = tokenizer.py_decode(tokens)
        print(f"✓ BPE训练后解码成功: {tokens} -> {decoded}")
        
        # 检查编码解码是否一致
        if test_text == decoded:
            print("✓ BPE训练后编码解码一致性测试通过")
            return True
        else:
            print(f"✗ BPE训练后编码解码不一致: 原文='{test_text}' 解码='{decoded}'")
            return False
            
    except Exception as e:
        print(f"✗ BPE分词器训练测试失败: {e}")
        return False

def test_bbpe_training():
    """测试BBPE分词器的训练功能"""
    try:
        from zero_tokenizer import BBPETokenizer
        
        # 创建分词器实例
        tokenizer = BBPETokenizer()
        
        # 准备训练数据
        training_texts = [
            "Hello world!",
            "Hello there!",
            "How are you?",
            "I am fine.",
            "Thank you!",
            "你好世界！",
            "你好吗？",
            "我很好，谢谢！",
            "不客气！",
            "再见！"
        ]
        
        # 训练分词器
        tokenizer.train(training_texts, 300)
        
        # 测试编码
        test_text = "Hello, 你好!"
        tokens = tokenizer.encode(test_text)
        print(f"✓ BBPE训练后编码成功: {test_text} -> {tokens}")
        
        # 测试解码
        decoded = tokenizer.decode(tokens)
        print(f"✓ BBPE训练后解码成功: {tokens} -> {decoded}")
        
        # 检查编码解码是否一致
        if test_text == decoded:
            print("✓ BBPE训练后编码解码一致性测试通过")
            return True
        else:
            print(f"✗ BBPE训练后编码解码不一致: 原文='{test_text}' 解码='{decoded}'")
            return False
            
    except Exception as e:
        print(f"✗ BBPE分词器训练测试失败: {e}")
        return False

def test_unigram_training():
    """测试Unigram分词器的训练功能"""
    try:
        from zero_tokenizer import UnigramTokenizer
        
        # 创建分词器实例
        tokenizer = UnigramTokenizer()
        
        # 准备训练数据
        training_texts = [
            "Hello world!",
            "Hello there!",
            "How are you?",
            "I am fine.",
            "Thank you!",
            "你好世界！",
            "你好吗？",
            "我很好，谢谢！",
            "不客气！",
            "再见！"
        ]
        
        # 训练分词器
        tokenizer.train(training_texts, 300)
        
        # 测试编码
        test_text = "Hello, 你好!"
        tokens = tokenizer.encode(test_text)
        print(f"✓ Unigram训练后编码成功: {test_text} -> {tokens}")
        
        # 测试解码
        decoded = tokenizer.decode(tokens)
        print(f"✓ Unigram训练后解码成功: {tokens} -> {decoded}")
        
        # 检查编码解码是否一致
        if test_text == decoded:
            print("✓ Unigram训练后编码解码一致性测试通过")
            return True
        else:
            print(f"✗ Unigram训练后编码解码不一致: 原文='{test_text}' 解码='{decoded}'")
            return False
            
    except Exception as e:
        print(f"✗ Unigram分词器训练测试失败: {e}")
        return False

def test_wordpiece_training():
    """测试WordPiece分词器的训练功能"""
    try:
        from zero_tokenizer import WordPieceTokenizer
        
        # 创建分词器实例
        tokenizer = WordPieceTokenizer()
        
        # 准备训练数据
        training_texts = [
            "Hello world!",
            "Hello there!",
            "How are you?",
            "I am fine.",
            "Thank you!",
            "你好世界！",
            "你好吗？",
            "我很好，谢谢！",
            "不客气！",
            "再见！"
        ]
        
        # 训练分词器
        tokenizer.train(training_texts, 300)
        
        # 测试编码
        test_text = "Hello, 你好!"
        tokens = tokenizer.encode(test_text)
        print(f"✓ WordPiece训练后编码成功: {test_text} -> {tokens}")
        
        # 测试解码
        decoded = tokenizer.decode(tokens)
        print(f"✓ WordPiece训练后解码成功: {tokens} -> {decoded}")
        
        # 检查编码解码是否一致
        if test_text == decoded:
            print("✓ WordPiece训练后编码解码一致性测试通过")
            return True
        else:
            print(f"✗ WordPiece训练后编码解码不一致: 原文='{test_text}' 解码='{decoded}'")
            return False
            
    except Exception as e:
        print(f"✗ WordPiece分词器训练测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始运行分词器训练测试...")
    print("=" * 50)
    
    tests = [
        test_bpe_training,
        test_bbpe_training,
        test_unigram_training,
        test_wordpiece_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # 空行分隔
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())