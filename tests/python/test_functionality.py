"""
测试分词器的编码和解码功能
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_bpe_tokenizer():
    """测试BPE分词器的基本功能"""
    try:
        from zero_tokenizer import Tokenizer
        
        # 创建分词器实例
        tokenizer = Tokenizer()
        
        # 测试编码
        text = "Hello, world! 你好，世界！"
        tokens = tokenizer.py_encode(text)
        print(f"✓ BPE编码成功: {text} -> {tokens}")
        
        # 测试解码
        decoded = tokenizer.py_decode(tokens)
        print(f"✓ BPE解码成功: {tokens} -> {decoded}")
        
        # 检查编码解码是否一致
        if text == decoded:
            print("✓ BPE编码解码一致性测试通过")
            return True
        else:
            print(f"✗ BPE编码解码不一致: 原文='{text}' 解码='{decoded}'")
            return False
            
    except Exception as e:
        print(f"✗ BPE分词器测试失败: {e}")
        return False

def test_bbp_e_tokenizer():
    """测试BBPE分词器的基本功能"""
    try:
        from zero_tokenizer import BBPETokenizer
        
        # 创建分词器实例
        tokenizer = BBPETokenizer()
        
        # 测试编码
        text = "Hello, world! 你好，世界！"
        tokens = tokenizer.encode(text)
        print(f"✓ BBPE编码成功: {text} -> {tokens}")
        
        # 测试解码
        decoded = tokenizer.decode(tokens)
        print(f"✓ BBPE解码成功: {tokens} -> {decoded}")
        
        # 检查编码解码是否一致
        if text == decoded:
            print("✓ BBPE编码解码一致性测试通过")
            return True
        else:
            print(f"✗ BBPE编码解码不一致: 原文='{text}' 解码='{decoded}'")
            return False
            
    except Exception as e:
        print(f"✗ BBPE分词器测试失败: {e}")
        return False

def test_unigram_tokenizer():
    """测试Unigram分词器的基本功能"""
    try:
        from zero_tokenizer import UnigramTokenizer
        
        # 创建分词器实例
        tokenizer = UnigramTokenizer()
        
        # 测试编码
        text = "Hello, world! 你好，世界！"
        tokens = tokenizer.encode(text)
        print(f"✓ Unigram编码成功: {text} -> {tokens}")
        
        # 测试解码
        decoded = tokenizer.decode(tokens)
        print(f"✓ Unigram解码成功: {tokens} -> {decoded}")
        
        # 检查编码解码是否一致
        if text == decoded:
            print("✓ Unigram编码解码一致性测试通过")
            return True
        else:
            print(f"✗ Unigram编码解码不一致: 原文='{text}' 解码='{decoded}'")
            return False
            
    except Exception as e:
        print(f"✗ Unigram分词器测试失败: {e}")
        return False

def test_wordpiece_tokenizer():
    """测试WordPiece分词器的基本功能"""
    try:
        from zero_tokenizer import WordPieceTokenizer
        
        # 创建分词器实例
        tokenizer = WordPieceTokenizer()
        
        # 测试编码
        text = "Hello, world! 你好，世界！"
        tokens = tokenizer.encode(text)
        print(f"✓ WordPiece编码成功: {text} -> {tokens}")
        
        # 测试解码
        decoded = tokenizer.decode(tokens)
        print(f"✓ WordPiece解码成功: {tokens} -> {decoded}")
        
        # 检查编码解码是否一致
        if text == decoded:
            print("✓ WordPiece编码解码一致性测试通过")
            return True
        else:
            print(f"✗ WordPiece编码解码不一致: 原文='{text}' 解码='{decoded}'")
            return False
            
    except Exception as e:
        print(f"✗ WordPiece分词器测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始运行分词器功能测试...")
    print("=" * 50)
    
    tests = [
        test_bpe_tokenizer,
        test_bbp_e_tokenizer,
        test_unigram_tokenizer,
        test_wordpiece_tokenizer
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