//! 测试辅助模块
//!
//! 提供通用的测试函数，减少各测试文件中的重复代码

use zero_tokenizer::prelude::*;

/// 通用测试函数：测试分词器的训练功能
pub fn test_tokenizer_training<T: Tokenizer>(tokenizer: &mut T, test_text: &str, vocab_size: u32) {
    // 测试未训练时的编码
    let tokens = tokenizer.encode("Hello world").unwrap();
    assert_eq!(tokens.len(), 11); // "Hello world"的UTF-8字节长度
    
    // 训练分词器
    tokenizer.train(vec![test_text.to_string()], vocab_size).unwrap();
    
    // 验证词汇表大小
    assert!(tokenizer.vocab_size() > 0);
    
    // 测试编码
    let tokens = tokenizer.encode(test_text).unwrap();
    assert!(!tokens.is_empty());
    
    // 测试解码
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, test_text);
}

/// 通用测试函数：测试编码解码无损性
pub fn test_encode_decode_roundtrip<T: Tokenizer>(tokenizer: &mut T, test_text: &str, vocab_size: u32) {
    // 训练分词器
    tokenizer.train(vec![test_text.to_string()], vocab_size).unwrap();
    
    // 编码
    let tokens = tokenizer.encode(test_text).unwrap();
    assert!(!tokens.is_empty());
    
    // 解码
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, test_text);
    
    // 测试空字符串
    let empty_tokens = tokenizer.encode("").unwrap();
    assert!(empty_tokens.is_empty());
    
    let empty_decoded = tokenizer.decode(&empty_tokens).unwrap();
    assert_eq!(empty_decoded, "");
}

/// 通用测试函数：测试默认构造
pub fn test_default_constructor<T: Tokenizer>(tokenizer: &T, expected_initial_size: usize) {
    assert_eq!(tokenizer.vocab_size(), expected_initial_size);
    
    // 未训练时编码应返回字节表示
    let text = "Hello";
    let tokens = tokenizer.encode(text).unwrap();
    // 对于已经初始化词汇表的分词器（如Unigram），每个字节可能对应一个token
    // 对于未初始化的分词器，每个字符对应一个token
    assert!(!tokens.is_empty());
}