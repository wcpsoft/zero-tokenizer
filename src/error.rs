use thiserror::Error;

/// 分词器错误类型
#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("编码错误: {message}")]
    EncodingError { message: String },

    #[error("解码错误: {message}")]
    DecodingError { message: String },

    #[error("训练错误: {message}")]
    TrainingError { message: String },

    #[error("模型加载错误: {message}")]
    ModelLoadError { message: String },

    #[error("模型保存错误: {message}")]
    ModelSaveError { message: String },

    #[error("词汇表错误: {message}")]
    VocabError { message: String },

    #[error("输入验证错误: {message}")]
    InputValidationError { message: String },

    #[error("初始化错误: {message}")]
    InitializationError { message: String },

    #[error("加载错误: {message}")]
    LoadError { message: String },

    #[error("分割错误: {message}")]
    SplitError { message: String },

    #[error("无效迭代器: {message}")]
    InvalidIterator { message: String },

    #[error("无效输入: {message}")]
    InvalidInput { message: String },

    #[error("无效正则表达式: {message}")]
    InvalidRegex { message: String },

    #[error("IO错误: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    #[error("序列化错误: {source}")]
    SerializationError {
        #[from]
        source: serde_json::Error,
    },

    #[error("正则表达式错误: {source}")]
    RegexError {
        #[from]
        source: fancy_regex::Error,
    },
}

/// 结果类型别名
pub type Result<T> = std::result::Result<T, TokenizerError>;

#[cfg(feature = "python")]
impl From<TokenizerError> for pyo3::PyErr {
    fn from(error: TokenizerError) -> Self {
        match error {
            TokenizerError::EncodingError { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            TokenizerError::DecodingError { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            TokenizerError::TrainingError { message } => {
                pyo3::exceptions::PyRuntimeError::new_err(message)
            }
            TokenizerError::ModelLoadError { message } => {
                pyo3::exceptions::PyIOError::new_err(message)
            }
            TokenizerError::ModelSaveError { message } => {
                pyo3::exceptions::PyIOError::new_err(message)
            }
            TokenizerError::VocabError { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            TokenizerError::InputValidationError { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            TokenizerError::InitializationError { message } => {
                pyo3::exceptions::PyRuntimeError::new_err(message)
            }
            TokenizerError::LoadError { message } => pyo3::exceptions::PyIOError::new_err(message),
            TokenizerError::SplitError { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            TokenizerError::InvalidIterator { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            TokenizerError::InvalidInput { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            TokenizerError::InvalidRegex { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            TokenizerError::IoError { source } => {
                pyo3::exceptions::PyIOError::new_err(source.to_string())
            }
            TokenizerError::SerializationError { source } => {
                pyo3::exceptions::PyValueError::new_err(source.to_string())
            }
            TokenizerError::RegexError { source } => {
                pyo3::exceptions::PyValueError::new_err(source.to_string())
            }
        }
    }
}

/// 创建编码错误
pub fn encoding_error(message: impl Into<String>) -> TokenizerError {
    TokenizerError::EncodingError {
        message: message.into(),
    }
}

/// 创建解码错误
pub fn decoding_error(message: impl Into<String>) -> TokenizerError {
    TokenizerError::DecodingError {
        message: message.into(),
    }
}

/// 创建训练错误
pub fn training_error(message: impl Into<String>) -> TokenizerError {
    TokenizerError::TrainingError {
        message: message.into(),
    }
}

/// 创建模型加载错误
pub fn model_load_error(message: impl Into<String>) -> TokenizerError {
    TokenizerError::ModelLoadError {
        message: message.into(),
    }
}

/// 创建模型保存错误
pub fn model_save_error(message: impl Into<String>) -> TokenizerError {
    TokenizerError::ModelSaveError {
        message: message.into(),
    }
}

/// 创建词汇表错误
pub fn vocab_error(message: impl Into<String>) -> TokenizerError {
    TokenizerError::VocabError {
        message: message.into(),
    }
}

/// 创建输入验证错误
pub fn input_validation_error(message: impl Into<String>) -> TokenizerError {
    TokenizerError::InputValidationError {
        message: message.into(),
    }
}
