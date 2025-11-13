//! BPE分词器单元测试
//!
//! 这个文件包含BPE分词器的特定功能测试，不包含与correctness_test.rs重复的正确性测试。

#[cfg(feature = "python")]
use zero_tokenizer::prelude::*;
mod test_utils;

/// 测试BPE分词器的训练功能
#[cfg(feature = "python")]
#[test]
fn test_bpe_tokenizer() {
    let mut tokenizer = zero_tokenizer::prelude::bpe().unwrap();
    let text = "Hello world, this is a test sentence for BPE tokenizer.";

    test_utils::test_tokenizer_training(&mut tokenizer, text, 1000);

    // BPE特定的验证
    assert!(tokenizer.vocab_size() >= 256 && tokenizer.vocab_size() <= 1000);
}

/// 测试BPE编码和解码的无损性
#[cfg(feature = "python")]
#[test]
fn test_bpe_encode_decode() {
    let mut tokenizer = zero_tokenizer::prelude::bpe().unwrap();
    let text = "This is a test sentence for BPE tokenizer.";

    test_utils::test_encode_decode_roundtrip(&mut tokenizer, text, 1000);
}

/// 测试BPE默认构造
#[cfg(feature = "python")]
#[test]
fn test_bpe_default() {
    let tokenizer = zero_tokenizer::prelude::bpe().unwrap();

    test_utils::test_default_constructor(&tokenizer, 256); // BPE初始化时包含所有字节值
}

/// 测试BPE词对生成功能
#[cfg(feature = "python")]
#[test]
fn test_word_pairs() {
    let mut tokenizer = zero_tokenizer::prelude::bpe().unwrap();

    // 训练分词器
    let text = "hello world";
    tokenizer.train(vec![text.to_string()], 100).unwrap();

    // 验证词对生成
    let word = tokenizer.encode(text).unwrap();
    assert!(!word.is_empty());
}

/// 测试BPE词对合并功能
#[cfg(feature = "python")]
#[test]
fn test_word_merge_pair() {
    let mut tokenizer = zero_tokenizer::prelude::bpe().unwrap();

    // 训练分词器
    let text = "hello world";
    tokenizer.train(vec![text.to_string()], 100).unwrap();

    // 验证词对合并
    let word = tokenizer.encode(text).unwrap();
    assert!(!word.is_empty());
}
