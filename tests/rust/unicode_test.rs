//! Unicode和特殊字符测试
//!
//! 测试各种Unicode字符的编码解码，包括emoji、组合字符、RTL文字等

use zero_tokenizer::prelude::*;

#[test]
fn test_emoji_encoding() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let text = "Hello 👋 World 🌍 Test 🎉";

    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_multiple_emojis() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let text = "😀😃😄😁😆😅🤣😂🙂🙃😉😊😇🥰😍🤩😘";

    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_combining_characters() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 使用组合字符: é 可以是 e + ´ (combining acute accent)
    let text = "café naïve";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_zero_width_characters() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 零宽字符
    let text = "Hello\u{200B}World\u{FEFF}Test"; // 零宽空格和零宽非断空格
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_rtl_text_arabic() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 阿拉伯文（从右到左）
    let text = "مرحبا بالعالم";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_rtl_text_hebrew() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 希伯来文（从右到左）
    let text = "שלום עולם";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_mixed_scripts() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 混合多种文字
    let text = "Hello 你好 مرحبا こんにちは 안녕하세요";
    tokenizer.train(vec![text.to_string()], 500).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_control_characters() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 控制字符
    let text = "Hello\nWorld\tTest\r\nDone";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_surrogate_pairs() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // Unicode代理对（4字节字符）- 数学字母
    let text = "𝕳𝖊𝖑𝖑𝖔 𝖂𝖔𝖗𝖑𝖉";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_cjk_unified_ideographs() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // CJK统一汉字
    let text = "中文测试 日本語テスト 한국어테스트";
    tokenizer.train(vec![text.to_string()], 400).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_special_punctuation() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 特殊标点符号 (使用原始字符串)
    let text = r#"‹›«»‚„''""…–—"#; // 各种引号、省略号、破折号
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_mathematical_symbols() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 数学符号
    let text = "∑∫√∞≠≤≥±×÷";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_currency_symbols() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 货币符号
    let text = "$€£¥₹₽₩¢";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_emoji_with_skin_tone() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 带肤色修饰符的emoji
    let text = "👋🏻👋🏼👋🏽👋🏾👋🏿";
    tokenizer.train(vec![text.to_string()], 300).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_unigram_unicode() {
    let mut tokenizer = zero_tokenizer::prelude::unigram().unwrap();

    // Unigram对各种Unicode的支持
    let text = "测试emoji👋和特殊字符€£¥";
    tokenizer.train(vec![text.to_string()], 16000).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}

#[test]
fn test_wordpiece_unicode() {
    let mut tokenizer = zero_tokenizer::prelude::wordpiece().unwrap();

    // WordPiece对各种Unicode的支持
    let text = "测试emoji🎉和数学符号∑∫";
    tokenizer.train(vec![text.to_string()], 16000).unwrap();

    let tokens = tokenizer.encode(text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();

    assert_eq!(text, decoded);
}
