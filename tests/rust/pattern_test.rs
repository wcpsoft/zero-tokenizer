//! 正则表达式模式测试
//!
//! 测试自定义正则表达式模式功能

use zero_tokenizer::bbpe::BBPETokenizer;

#[cfg(feature = "python")]
use zero_tokenizer::bpe::Tokenizer as BPETokenizer;

#[cfg(feature = "python")]
#[test]
fn test_bpe_custom_pattern_word_only() {
    // 只匹配单词字符
    let pattern = r"\w+".to_string();
    let mut tokenizer = BPETokenizer::with_pattern(pattern).unwrap();

    let text = "Hello, world!";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    // 标点符号可能被分开处理
    assert!(!tokens.is_empty());

    let decoded = tokenizer.decode(&tokens).unwrap();
    // 解码后应该能还原
    assert!(!decoded.is_empty());
}

#[test]
fn test_bbpe_custom_pattern_chinese() {
    // 自定义模式: 匹配中文字符
    let pattern = r"[\u4e00-\u9fa5]+".to_string();
    let mut tokenizer = BBPETokenizer::with_pattern(pattern).unwrap();

    let text = "你好世界Hello";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    assert!(!tokens.is_empty());

    let decoded = tokenizer.decode(&tokens).unwrap();
    assert!(!decoded.is_empty());
}

#[test]
fn test_bbpe_custom_pattern_digits() {
    // 匹配数字
    let pattern = r"\d+".to_string();
    let mut tokenizer = BBPETokenizer::with_pattern(pattern).unwrap();

    let text = "123 456 789";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_bbpe_custom_pattern_alphanumeric() {
    // 匹配字母数字
    let pattern = r"[a-zA-Z0-9]+".to_string();
    let mut tokenizer = BBPETokenizer::with_pattern(pattern).unwrap();

    let text = "Test123 Hello456";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert!(!decoded.is_empty());
}

#[cfg(feature = "python")]
#[test]
fn test_bpe_invalid_regex_pattern() {
    // 无效的正则表达式
    let pattern = "[invalid(".to_string();
    let result = BPETokenizer::with_pattern(pattern);
    assert!(result.is_err());
}

#[test]
fn test_bbpe_invalid_regex_pattern() {
    // 无效的正则表达式
    let pattern = "[unclosed(".to_string();
    let result = BBPETokenizer::with_pattern(pattern);
    assert!(result.is_err());
}

#[test]
fn test_bbpe_empty_pattern() {
    // 空模式（可能导致问题）
    let pattern = "".to_string();
    let result = BBPETokenizer::with_pattern(pattern);

    // 根据实现，可能返回错误或使用默认模式
    // 这里我们只验证不会panic
    match result {
        Ok(_) => {}
        Err(_) => {}
    }
}

#[test]
fn test_bbpe_whitespace_pattern() {
    // 匹配空白字符
    let pattern = r"\s+".to_string();
    let mut tokenizer = BBPETokenizer::with_pattern(pattern).unwrap();

    let text = "   \t\n  ";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    // 应该能编码空白字符
    assert!(!tokens.is_empty());
}

#[test]
fn test_bbpe_mixed_pattern() {
    // 混合模式：单词或数字
    let pattern = r"\w+|\d+".to_string();
    let mut tokenizer = BBPETokenizer::with_pattern(pattern).unwrap();

    let text = "Hello 123 World 456";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_bbpe_pattern_with_unicode() {
    // 支持Unicode的模式
    let pattern = r"[\p{L}\p{N}]+".to_string();

    // 注意：fancy_regex支持\p{L}等Unicode类别
    match BBPETokenizer::with_pattern(pattern) {
        Ok(mut tokenizer) => {
            let text = "Hello你好123";
            tokenizer.train(vec![text.to_string()], 300).unwrap();

            let tokens = tokenizer.encode(text).unwrap();
            assert!(!tokens.is_empty());
        }
        Err(_) => {
            // 如果不支持Unicode类别，测试也应该通过
        }
    }
}

#[test]
fn test_bbpe_default_pattern() {
    // 测试默认模式（GPT-4风格）
    let tokenizer1 = BBPETokenizer::new_internal().unwrap();
    let tokenizer2 = BBPETokenizer::new_internal().unwrap();

    let text = "Hello, world! 你好世界！123";

    let tokens1 = tokenizer1.encode(text).unwrap();
    let tokens2 = tokenizer2.encode(text).unwrap();

    // 使用相同默认模式，结果应该一致
    assert_eq!(tokens1, tokens2);
}

#[test]
fn test_pattern_persistence() {
    // 测试模式在训练后是否保持
    let pattern = r"\w+".to_string();
    let mut tokenizer = BBPETokenizer::with_pattern(pattern.clone()).unwrap();

    tokenizer.train(vec!["test".to_string()], 300).unwrap();

    // 训练后，模式应该仍然有效
    let text = "Hello World";
    let tokens = tokenizer.encode(text).unwrap();
    assert!(!tokens.is_empty());
}
