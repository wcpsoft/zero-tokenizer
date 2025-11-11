//! WordPiece分词器单元测试
//!
//! 这个文件包含WordPiece分词器的特定功能测试，不包含与correctness_test.rs重复的正确性测试。

use zero_tokenizer::prelude::*;
mod test_utils;

/// 测试WordPiece分词器的训练功能
#[test]
fn test_wordpiece_tokenizer() {
    let text = "这是一个测试文本，用于验证分词器的功能。";
    let mut tokenizer = zero_tokenizer::prelude::wordpiece().unwrap();
    
    // 训练分词器，指定一个大于初始词汇表大小的值
    tokenizer.train(vec![text.to_string()], 16000).unwrap();
    
    // 验证词汇表大小 - 训练后的词汇表大小应该大于初始大小
    assert!(tokenizer.vocab_size() > 256 + 15001); // 大于字节标记+常用汉字
    
    // 测试编码
    let tokens = tokenizer.encode(text).unwrap();
    assert!(!tokens.is_empty());
    
    // 测试解码
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

/// 测试WordPiece分词功能
#[test]
fn test_wordpiece_tokenize() {
    let mut tokenizer = zero_tokenizer::prelude::wordpiece().unwrap();
    let text = "Hello world, this is a test sentence for WordPiece tokenizer.";
    
    // 训练分词器
    tokenizer.train(vec![text.to_string()], 1000).unwrap();
    
    // 测试分词
    let tokens = tokenizer.encode(text).unwrap();
    assert!(!tokens.is_empty());
    
    // 测试解码
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

/// 测试WordPiece编码和解码功能
#[test]
fn test_wordpiece_encode_decode() {
    let mut tokenizer = zero_tokenizer::prelude::wordpiece().unwrap();
    let text = "This is a test sentence for WordPiece tokenizer.";
    
    test_utils::test_encode_decode_roundtrip(&mut tokenizer, text, 1000);
}

/// 测试WordPiece默认构造
#[test]
fn test_wordpiece_default() {
    // WordPiece特定的验证 - 初始词汇表大小应为256+15001
    let tokenizer = zero_tokenizer::prelude::wordpiece().unwrap();
    
    // WordPiece特定的验证 - 初始词汇表大小应为256+15001
    assert_eq!(tokenizer.vocab_size(), 256 + 15001); // 256个字节 + 15001个常用汉字
}