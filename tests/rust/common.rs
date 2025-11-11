/// 共享的测试样本和工具函数
pub const TEST_SAMPLES: [&str; 10] = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Rust is a systems programming language",
    "Machine learning and artificial intelligence",
    "Natural language processing with transformers",
    "Byte Pair Encoding for tokenization",
    "Unigram models are probabilistic",
    "WordPiece algorithm used in BERT",
    "This is a test sentence for tokenization",
    "Programming in Rust is fun and efficient"
];

/// 边缘测试用例
pub const EDGE_CASES: [&str; 8] = [
    "", " ", "a", "ab", "abc", "Hello", "Hello world", "Hello world!"
];

/// 长文本测试用例
pub const LONG_TEXT: &str = "This is a very long text sample that is used to test the tokenizers with longer sequences. \
                             It contains multiple sentences and various types of words to ensure that the tokenizers \
                             can handle longer inputs properly. The text includes punctuation, numbers like 12345, \
                             and special characters such as @#$%^&*() to test robustness.";

/// 创建测试文本集合
pub fn create_test_texts() -> Vec<String> {
    TEST_SAMPLES.iter().map(|&s| s.to_string()).collect()
}

/// 创建边缘测试文本集合
pub fn create_edge_texts() -> Vec<String> {
    EDGE_CASES.iter().map(|&s| s.to_string()).collect()
}