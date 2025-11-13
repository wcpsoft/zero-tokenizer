//! 前导模块
//!
//! 导出所有常用的类型和特征，方便使用。

pub use crate::base::traits::{SubwordTokenizer, Tokenizer};
pub use crate::bbpe::BBPETokenizer as BBPE;
#[cfg(feature = "python")]
pub use crate::bpe::Tokenizer as BPE;
pub use crate::unigram::UnigramTokenizer as Unigram;
pub use crate::wordpiece::WordPieceTokenizer as WordPiece;

/// 创建BPE分词器的便捷函数
#[cfg(feature = "python")]
pub fn bpe() -> Result<BPE, String> {
    BPE::_new_internal()
}

/// 创建BBPE分词器的便捷函数
pub fn bbpe() -> Result<BBPE, String> {
    BBPE::new_internal()
}

/// 创建Unigram分词器的便捷函数
pub fn unigram() -> Result<Unigram, String> {
    Unigram::new_internal()
}

/// 创建WordPiece分词器的便捷函数
pub fn wordpiece() -> Result<WordPiece, String> {
    WordPiece::new_internal()
}
