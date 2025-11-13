//! 错误处理测试
//!
//! 测试所有错误路径和异常情况

use zero_tokenizer::prelude::*;

#[test]
fn test_decode_invalid_token_id() {
    let tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 使用一个不存在的token ID
    let invalid_tokens = vec![999999];
    let result = tokenizer.decode(&invalid_tokens);

    assert!(result.is_err());
    if let Err(e) = result {
        // 验证错误消息包含相关信息
        assert!(e.contains("未找到") || e.contains("not found") || e.contains("ID"));
    }
}

#[test]
fn test_train_with_invalid_vocab_size() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // BBPE需要至少256的词汇表大小
    let result = tokenizer.train(vec!["test".to_string()], 100);
    assert!(result.is_err());

    if let Err(e) = result {
        assert!(e.contains("256") || e.contains("至少") || e.contains("vocab"));
    }
}

#[test]
fn test_train_with_empty_texts() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 空训练数据
    let result = tokenizer.train(vec![], 300);

    // 根据实际实现，可能成功但不产生新token，或返回错误
    // 这里我们接受两种结果
    match result {
        Ok(_) => {
            // 如果成功，词汇表应该还是初始大小
            assert_eq!(tokenizer.vocab_size(), 256);
        }
        Err(_) => {
            // 如果失败也是合理的
        }
    }
}

#[test]
fn test_encode_empty_string() {
    let tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let result = tokenizer.encode("");

    assert!(result.is_ok());
    let tokens = result.unwrap();
    assert!(tokens.is_empty());
}

#[test]
fn test_decode_empty_tokens() {
    let tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    let result = tokenizer.decode(&[]);

    assert!(result.is_ok());
    let text = result.unwrap();
    assert_eq!(text, "");
}

#[cfg(feature = "python")]
#[test]
fn test_bpe_invalid_pattern() {
    use zero_tokenizer::bpe::Tokenizer;

    // 无效的正则表达式
    let result = Tokenizer::with_pattern("[invalid(".to_string());
    assert!(result.is_err());

    if let Err(e) = result {
        assert!(e.contains("正则") || e.contains("regex") || e.contains("pattern"));
    }
}

#[test]
fn test_bbpe_invalid_pattern() {
    use zero_tokenizer::bbpe::BBPETokenizer;

    // 无效的正则表达式
    let result = BBPETokenizer::with_pattern("[unclosed(".to_string());
    assert!(result.is_err());
}

#[test]
fn test_decode_partially_invalid_tokens() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();
    tokenizer.train(vec!["test".to_string()], 300).unwrap();

    // 混合有效和无效的token ID
    let mixed_tokens = vec![0, 1, 999999, 2];
    let result = tokenizer.decode(&mixed_tokens);

    // 应该在遇到第一个无效ID时失败
    assert!(result.is_err());
}

#[test]
fn test_train_with_single_empty_string() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 单个空字符串
    let result = tokenizer.train(vec!["".to_string()], 300);

    // 应该成功或返回合理的错误
    match result {
        Ok(_) => {
            // 词汇表应该是初始大小
            assert_eq!(tokenizer.vocab_size(), 256);
        }
        Err(_) => {
            // 也可以接受错误
        }
    }
}

#[test]
fn test_unigram_decode_invalid_token() {
    let tokenizer = zero_tokenizer::prelude::unigram().unwrap();

    // 使用超出词汇表范围的ID
    let invalid_tokens = vec![9999999];
    let result = tokenizer.decode(&invalid_tokens);

    assert!(result.is_err());
}

#[test]
fn test_wordpiece_decode_invalid_token() {
    let tokenizer = zero_tokenizer::prelude::wordpiece().unwrap();

    // 使用超出词汇表范围的ID
    let invalid_tokens = vec![9999999];
    let result = tokenizer.decode(&invalid_tokens);

    assert!(result.is_err());
}

#[test]
fn test_train_with_very_small_vocab_size() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 词汇表大小为1（远小于256）
    let result = tokenizer.train(vec!["test".to_string()], 1);
    assert!(result.is_err());
}

#[test]
fn test_encode_after_failed_training() {
    let mut tokenizer = zero_tokenizer::prelude::bbpe().unwrap();

    // 尝试失败的训练
    let _ = tokenizer.train(vec!["test".to_string()], 100);

    // 应该仍能编码（使用初始词汇表）
    let result = tokenizer.encode("test");
    assert!(result.is_ok());
    assert!(!result.unwrap().is_empty());
}
