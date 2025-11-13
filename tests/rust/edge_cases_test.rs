//! 边界情况测试
//!
//! 测试各种边界情况和极端输入

use zero_tokenizer::prelude::*;

#[test]
fn test_single_character() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    tokenizer.train(vec!["a".to_string()], 300).unwrap();

    let tokens = tokenizer.encode("a").unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, "a");
}

#[test]
fn test_single_byte() {
    let tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 单个ASCII字符
    let tokens = tokenizer.encode("x").unwrap();
    assert_eq!(tokens.len(), 1);

    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, "x");
}

#[test]
fn test_very_long_text() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 10KB文本
    let long_text = "a".repeat(10_000);
    tokenizer.train(vec![long_text.clone()], 300).unwrap();

    let tokens = tokenizer.encode(&long_text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, long_text);
}

#[test]
fn test_repeated_training() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 第一次训练
    tokenizer.train(vec!["Hello".to_string()], 300).unwrap();
    let vocab_size_1 = tokenizer.vocab_size();

    // 第二次训练（应该重置并重新训练）
    tokenizer.train(vec!["World".to_string()], 300).unwrap();
    let vocab_size_2 = tokenizer.vocab_size();

    // 验证第二次训练生效
    assert!(vocab_size_2 >= 256);

    // 可以成功编码第二次训练的文本
    let tokens = tokenizer.encode("World").unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_whitespace_only() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    let text = "   \t\n  ";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_numbers_and_symbols() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    let text = "12345 + 67890 = 80235 @#$%^&*()";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_repeated_characters() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 重复字符
    let text = "aaaaaaaaaa bbbbbbbbb cccccccc";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_minimum_vocab_size() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 使用最小允许的词汇表大小
    tokenizer.train(vec!["test".to_string()], 256).unwrap();

    assert_eq!(tokenizer.vocab_size(), 256);

    // 仍然可以正常编码解码
    let tokens = tokenizer.encode("test").unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, "test");
}

#[test]
fn test_large_vocab_size() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 使用较大的词汇表
    tokenizer
        .train(vec!["test text".to_string()], 1000)
        .unwrap();

    // 词汇表大小应该在合理范围内
    assert!(tokenizer.vocab_size() >= 256);
    assert!(tokenizer.vocab_size() <= 1000);
}

#[test]
fn test_all_punctuation() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    let text = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_mixed_case() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    let text = "HeLLo WoRLd TeST";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_newlines_and_tabs() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    let text = "line1\nline2\tline3\r\nline4";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_unigram_single_character() {
    let mut tokenizer = zero_tokenizer::prelude::unigram().unwrap();

    // Unigram初始化时已有大量常用汉字
    let text = "测";
    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_wordpiece_single_character() {
    let mut tokenizer = zero_tokenizer::prelude::wordpiece().unwrap();

    // WordPiece初始化时已有大量常用汉字
    let text = "测";
    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_encode_decode_consistency() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    let texts = vec![
        "a",
        "ab",
        "abc",
        "Hello",
        "Hello World",
        "测试",
        "123",
        "!@#",
        "",
    ];

    for text in texts {
        tokenizer.train(vec![text.to_string()], 300).unwrap();

        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert_eq!(
            decoded, text,
            "编码解码不一致: '{}' -> {:?} -> '{}'",
            text, tokens, decoded
        );
    }
}

#[test]
fn test_binary_like_text() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 类似二进制的文本
    let text = "\0\x01\x02\x03\x04\x05";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}
