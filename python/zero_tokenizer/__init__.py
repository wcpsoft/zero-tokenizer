"""
Zero Tokenizer - A fast tokenizer implementation in Rust with Python bindings.

This package provides various tokenization algorithms including:
- BPE (Byte Pair Encoding)
- BBPE (Byte-level BPE)
- Unigram Language Model
- WordPiece
"""

from ._zero_tokenizer import Tokenizer, BBPETokenizer, UnigramTokenizer, WordPieceTokenizer

__version__ = "0.1.0"
__all__ = ["Tokenizer", "BBPETokenizer", "UnigramTokenizer", "WordPieceTokenizer"]

# 为了向后兼容，创建别名
BPETokenizer = Tokenizer