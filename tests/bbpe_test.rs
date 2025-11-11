//! BBPE分词器单元测试
//!
//! 这个文件包含BBPE分词器的特定功能测试，不包含与correctness_test.rs重复的正确性测试。

use zero_tokenizer::prelude::*;
mod test_utils;

/// 测试BBPE分词器的训练功能
#[test]
fn test_bbpe_tokenizer() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let text = "Hello world, this is a test sentence for BBPE tokenizer.";
    
    test_utils::test_tokenizer_training(&mut tokenizer, text, 300);
    
    // BBPE特定的验证
    assert!(tokenizer.vocab_size() >= 256 && tokenizer.vocab_size() <= 300);
}

/// 测试BBPE编码和解码的无损性
#[test]
fn test_bbpe_encode_decode() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let text = "This is a test sentence for BBPE tokenizer.";
    
    test_utils::test_encode_decode_roundtrip(&mut tokenizer, text, 300);
}

/// 测试BBPE默认构造
#[test]
fn test_bbpe_default() {
    let tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    
    test_utils::test_default_constructor(&tokenizer, 256); // BBPE初始化时包含所有字节值
}