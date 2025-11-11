#!/usr/bin/env python3
"""
简单的Python测试，验证zero_tokenizer模块是否可以正确导入和使用
"""

import sys

def test_import():
    """测试模块导入"""
    try:
        import zero_tokenizer
        print("✓ zero_tokenizer模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ zero_tokenizer模块导入失败: {e}")
        return False

def test_tokenizer_classes():
    """测试分词器类是否可用"""
    try:
        from zero_tokenizer import Tokenizer, BBPETokenizer, UnigramTokenizer, WordPieceTokenizer
        print("✓ 所有分词器类导入成功")
        return True
    except ImportError as e:
        print(f"✗ 分词器类导入失败: {e}")
        return False

def test_tokenizer_creation():
    """测试分词器实例创建"""
    try:
        from zero_tokenizer import Tokenizer
        tokenizer = Tokenizer()
        print("✓ Tokenizer实例创建成功")
        return True
    except Exception as e:
        print(f"✗ Tokenizer实例创建失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始运行zero_tokenizer基本测试...")
    print("=" * 50)
    
    tests = [
        test_import,
        test_tokenizer_classes,
        test_tokenizer_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
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