//! 保存和加载功能测试
//!
//! 测试所有分词器的模型持久化功能

use std::fs;
use std::path::Path;
use zero_tokenizer::prelude::*;

/// 清理测试文件的辅助函数
fn cleanup_test_file(path: &str) {
    if Path::new(path).exists() {
        fs::remove_file(path).ok();
    }
}

#[cfg(feature = "python")]
#[test]
fn test_bpe_save_load_roundtrip() {
    let model_path = "test_bpe.model";
    cleanup_test_file(model_path);

    // 训练并保存
    let mut tokenizer = zero_tokenizer::prelude::bpe().unwrap();
    let training_text = "Hello world! This is a test.";
    tokenizer
        .train(vec![training_text.to_string()], 300)
        .unwrap();

    let original_vocab_size = tokenizer.vocab_size();
    let original_tokens = tokenizer.encode(training_text).unwrap();

    tokenizer.save(model_path).unwrap();

    // 加载并验证
    let mut loaded = zero_tokenizer::prelude::bpe().unwrap();
    loaded.load(model_path).unwrap();

    assert_eq!(loaded.vocab_size(), original_vocab_size);

    let loaded_tokens = loaded.encode(training_text).unwrap();
    assert_eq!(loaded_tokens, original_tokens);

    let decoded = loaded.decode(&loaded_tokens).unwrap();
    assert_eq!(decoded, training_text);

    cleanup_test_file(model_path);
}

#[test]
fn test_bbpe_save_load_roundtrip() {
    let model_path = "test_bbpe.model";
    cleanup_test_file(model_path);

    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let training_text = "Hello world! 你好世界！";
    tokenizer
        .train(vec![training_text.to_string()], 300)
        .unwrap();

    let original_vocab_size = tokenizer.vocab_size();
    let original_tokens = tokenizer.encode(training_text).unwrap();

    tokenizer.save(model_path).unwrap();

    let mut loaded = zero_tokenizer::prelude::bbpe().unwrap();
    loaded.load(model_path).unwrap();

    assert_eq!(loaded.vocab_size(), original_vocab_size);

    let loaded_tokens = loaded.encode(training_text).unwrap();
    assert_eq!(loaded_tokens, original_tokens);

    let decoded = loaded.decode(&loaded_tokens).unwrap();
    assert_eq!(decoded, training_text);

    cleanup_test_file(model_path);
}

#[test]
fn test_unigram_save_load_roundtrip() {
    let model_path = "test_unigram.model";
    cleanup_test_file(model_path);

    let mut tokenizer = zero_tokenizer::prelude::unigram().unwrap();
    let training_text = "这是一个测试文本";
    tokenizer
        .train(vec![training_text.to_string()], 16000)
        .unwrap();

    let original_vocab_size = tokenizer.vocab_size();
    let original_tokens = tokenizer.encode(training_text).unwrap();

    tokenizer.save(model_path).unwrap();

    let mut loaded = zero_tokenizer::prelude::unigram().unwrap();
    loaded.load(model_path).unwrap();

    assert_eq!(loaded.vocab_size(), original_vocab_size);

    let loaded_tokens = loaded.encode(training_text).unwrap();
    assert_eq!(loaded_tokens, original_tokens);

    let decoded = loaded.decode(&loaded_tokens).unwrap();
    assert_eq!(decoded, training_text);

    cleanup_test_file(model_path);
}

#[test]
fn test_wordpiece_save_load_roundtrip() {
    let model_path = "test_wordpiece.model";
    cleanup_test_file(model_path);

    let mut tokenizer = zero_tokenizer::prelude::wordpiece().unwrap();
    let training_text = "这是一个测试文本";
    tokenizer
        .train(vec![training_text.to_string()], 16000)
        .unwrap();

    let original_vocab_size = tokenizer.vocab_size();
    let original_tokens = tokenizer.encode(training_text).unwrap();

    tokenizer.save(model_path).unwrap();

    let mut loaded = zero_tokenizer::prelude::wordpiece().unwrap();
    loaded.load(model_path).unwrap();

    assert_eq!(loaded.vocab_size(), original_vocab_size);

    let loaded_tokens = loaded.encode(training_text).unwrap();
    assert_eq!(loaded_tokens, original_tokens);

    let decoded = loaded.decode(&loaded_tokens).unwrap();
    assert_eq!(decoded, training_text);

    cleanup_test_file(model_path);
}

#[test]
fn test_save_to_invalid_path() {
    let tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let result = tokenizer.save("/invalid/nonexistent/path/model.bin");
    assert!(result.is_err());
}

#[test]
fn test_load_nonexistent_file() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let result = tokenizer.load("nonexistent_file_12345.model");
    assert!(result.is_err());
}

#[test]
fn test_load_corrupted_file() {
    let model_path = "corrupted.model";
    fs::write(model_path, b"invalid corrupted data").unwrap();

    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let result = tokenizer.load(model_path);

    // 应该返回错误，因为文件格式不正确
    assert!(result.is_err());

    cleanup_test_file(model_path);
}

#[test]
fn test_save_multiple_times() {
    let model_path = "test_multiple_saves.model";
    cleanup_test_file(model_path);

    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    tokenizer.train(vec!["test".to_string()], 300).unwrap();

    // 第一次保存
    tokenizer.save(model_path).unwrap();

    // 再次训练
    tokenizer
        .train(vec!["another test".to_string()], 300)
        .unwrap();

    // 第二次保存（应该覆盖）
    tokenizer.save(model_path).unwrap();

    // 验证加载的是最新版本
    let mut loaded = zero_tokenizer::prelude::bbpe().unwrap();
    loaded.load(model_path).unwrap();

    assert_eq!(loaded.vocab_size(), tokenizer.vocab_size());

    cleanup_test_file(model_path);
}
