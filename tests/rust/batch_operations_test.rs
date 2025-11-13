//! 批量操作测试
//!
//! 测试并行编码解码功能

use rayon::prelude::*;
use zero_tokenizer::prelude::*;

#[test]
fn test_bbpe_parallel_encode() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let texts = vec![
        "Hello world!".to_string(),
        "你好世界！".to_string(),
        "Test text".to_string(),
        "Another test".to_string(),
    ];

    tokenizer.train(texts.clone(), 300).unwrap();

    // 并行编码
    let batch_results: Vec<Vec<u32>> = texts
        .par_iter()
        .map(|text| tokenizer.encode(text).unwrap())
        .collect();

    assert_eq!(batch_results.len(), texts.len());

    // 验证每个结果
    for (text, tokens) in texts.iter().zip(batch_results.iter()) {
        let decoded = tokenizer.decode(tokens).unwrap();
        assert_eq!(*text, decoded);
    }
}

#[test]
fn test_bbpe_parallel_decode() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let texts = vec!["Hello world!".to_string(), "你好世界！".to_string()];

    tokenizer.train(texts.clone(), 300).unwrap();

    // 编码所有文本
    let token_lists: Vec<Vec<u32>> = texts
        .iter()
        .map(|text| tokenizer.encode(text).unwrap())
        .collect();

    // 并行解码
    let batch_results: Vec<String> = token_lists
        .par_iter()
        .map(|tokens| tokenizer.decode(tokens).unwrap())
        .collect();

    // 验证
    assert_eq!(batch_results, texts);
}

#[test]
fn test_large_batch_encode() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    tokenizer.train(vec!["test".to_string()], 300).unwrap();

    // 100个文本
    let texts: Vec<String> = (0..100)
        .map(|i| format!("Test text number {}", i))
        .collect();

    // 并行批量处理
    let results: Vec<Vec<u32>> = texts
        .par_iter()
        .map(|text| tokenizer.encode(text).unwrap())
        .collect();

    assert_eq!(results.len(), 100);

    // 验证所有结果都非空
    for tokens in &results {
        assert!(!tokens.is_empty());
    }
}

#[test]
fn test_unigram_parallel_encode() {
    let mut tokenizer = zero_tokenizer::prelude::unigram().unwrap();
    let texts = vec![
        "测试文本1".to_string(),
        "测试文本2".to_string(),
        "测试文本3".to_string(),
    ];

    tokenizer.train(texts.clone(), 16000).unwrap();

    // 并行编码
    let batch_results: Vec<Vec<u32>> = texts
        .par_iter()
        .map(|text| tokenizer.encode(text).unwrap())
        .collect();

    assert_eq!(batch_results.len(), texts.len());

    // 验证解码一致性
    for (original, tokens) in texts.iter().zip(batch_results.iter()) {
        let decoded = tokenizer.decode(tokens).unwrap();
        assert_eq!(*original, decoded);
    }
}

#[test]
fn test_wordpiece_parallel_encode() {
    let mut tokenizer = zero_tokenizer::prelude::wordpiece().unwrap();
    let texts = vec![
        "测试文本1".to_string(),
        "测试文本2".to_string(),
        "测试文本3".to_string(),
    ];

    tokenizer.train(texts.clone(), 16000).unwrap();

    // 并行编码
    let batch_results: Vec<Vec<u32>> = texts
        .par_iter()
        .map(|text| tokenizer.encode(text).unwrap())
        .collect();

    assert_eq!(batch_results.len(), texts.len());

    // 验证解码一致性
    for (original, tokens) in texts.iter().zip(batch_results.iter()) {
        let decoded = tokenizer.decode(tokens).unwrap();
        assert_eq!(*original, decoded);
    }
}

#[cfg(feature = "python")]
#[test]
fn test_bpe_parallel_encode() {
    let mut tokenizer = zero_tokenizer::prelude::bpe().unwrap();
    let texts = vec![
        "Hello world!".to_string(),
        "Test text".to_string(),
        "Another example".to_string(),
    ];

    tokenizer.train(texts.clone(), 300).unwrap();

    // 并行编码
    let batch_results: Vec<Vec<u32>> = texts
        .par_iter()
        .map(|text| tokenizer.encode(text).unwrap())
        .collect();

    assert_eq!(batch_results.len(), texts.len());
}

#[test]
fn test_mixed_length_texts() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 不同长度的文本
    let texts = vec![
        "a".to_string(),
        "ab".to_string(),
        "abc def".to_string(),
        "This is a much longer text for testing".to_string(),
        "x".repeat(1000), // 长文本
    ];

    tokenizer.train(texts.clone(), 300).unwrap();

    // 并行处理
    let batch_results: Vec<Vec<u32>> = texts
        .par_iter()
        .map(|text| tokenizer.encode(text).unwrap())
        .collect();

    // 验证
    for (original, tokens) in texts.iter().zip(batch_results.iter()) {
        let decoded = tokenizer.decode(tokens).unwrap();
        assert_eq!(*original, decoded);
    }
}

#[test]
fn test_empty_texts_in_batch() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    tokenizer.train(vec!["test".to_string()], 300).unwrap();

    // 包含空字符串的批次
    let texts = vec![
        "Hello".to_string(),
        "".to_string(),
        "World".to_string(),
        "".to_string(),
    ];

    // 并行编码
    let batch_results: Vec<Vec<u32>> = texts
        .par_iter()
        .map(|text| tokenizer.encode(text).unwrap())
        .collect();

    // 验证空字符串编码为空token列表
    assert!(!batch_results[0].is_empty()); // "Hello"
    assert!(batch_results[1].is_empty()); // ""
    assert!(!batch_results[2].is_empty()); // "World"
    assert!(batch_results[3].is_empty()); // ""
}
