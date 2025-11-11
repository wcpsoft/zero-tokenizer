#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod base;
pub mod bpe;
pub mod bbpe;
pub mod unigram;
pub mod wordpiece;
pub mod prelude;

/// 导出所有分词器到Python
#[cfg(feature = "python")]
#[pymodule]
fn zero_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // forwards Rust `log` to Python's `logging`
    m.add_class::<bpe::bpe::Tokenizer>()?;
    m.add_class::<bbpe::bbpe::BBPETokenizer>()?;
    m.add_class::<unigram::unigram::UnigramTokenizer>()?;
    m.add_class::<wordpiece::wordpiece::WordPieceTokenizer>()?;
    Ok(())
}