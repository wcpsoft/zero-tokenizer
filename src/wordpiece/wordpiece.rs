use std::collections::HashMap;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

use crate::base::traits::{Tokenizer, SubwordTokenizer};
use crate::base::tokenizer_base::TokenizerBase;

/// WordPiece分词器
#[cfg_attr(feature = "python", pyclass)]
pub struct WordPieceTokenizer {
    /// 基础分词器
    pub base: TokenizerBase<u32>,
    /// 标记分数
    pub scores: Vec<f64>,
    /// 未知标记ID
    pub unk_token_id: u32,
}

impl WordPieceTokenizer {
    /// 创建新的WordPiece分词器
    pub fn new_internal() -> Result<Self, String> {
        let base = TokenizerBase::new()?;
        
        let mut tokenizer = Self {
            base,
            scores: Vec::new(),
            unk_token_id: 0,
        };
        
        // 初始化字节词汇表和常用汉字
        tokenizer.init_byte_vocab();
        tokenizer.load_common_chinese_chars()?;
        
        Ok(tokenizer)
    }

    /// 使用自定义正则表达式模式创建新的WordPiece分词器
    pub fn with_pattern_internal(pattern: String) -> Result<Self, String> {
        let base = TokenizerBase::with_pattern(pattern)?;
        
        let mut tokenizer = Self {
            base,
            scores: Vec::new(),
            unk_token_id: 0,
        };
        
        // 初始化字节词汇表和常用汉字
        tokenizer.init_byte_vocab();
        tokenizer.load_common_chinese_chars()?;
        
        Ok(tokenizer)
    }

    /// 初始化词汇表，添加所有字节值
    fn init_byte_vocab(&mut self) {
        // 清空现有词汇表
        self.base.vocab.clear();
        self.base.vocab_rev.clear();
        self.scores.clear();
        
        // 添加所有字节值
        for i in 0..=255 {
            let byte_str = format!("<0x{:02X}>", i);
            let _byte_vec = vec![i as u8];
            
            // 将字节向量转换为字符串表示
            let token = if i >= 32 && i <= 126 {
                // 可打印ASCII字符
                char::from(i).to_string()
            } else {
                // 非打印字符使用特殊表示
                byte_str
            };
            
            self.base.vocab.insert(token.clone(), i as u32);
            self.base.vocab_rev.insert(i as u32, token);
            self.scores.push(0.0); // 初始分数为0
        }
        
        // 设置未知标记ID
        self.unk_token_id = 0;
    }
    
    /// 加载常用汉字
    fn load_common_chinese_chars(&mut self) -> Result<(), String> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        
        let file_path = "dict/常用汉字字表.txt";
        let file = File::open(file_path)
            .map_err(|e| format!("无法打开常用汉字文件: {}", e))?;
        
        let reader = BufReader::new(file);
        
        for line in reader.lines() {
            let line = line.map_err(|e| format!("读取常用汉字文件失败: {}", e))?;
            let char_str = line.trim();
            if !char_str.is_empty() {
                // 将汉字添加到词汇表
                let token_id = self.base.vocab.len() as u32;
                self.base.vocab.insert(char_str.to_string(), token_id);
                self.base.vocab_rev.insert(token_id, char_str.to_string());
                self.scores.push(0.0); // 初始分数为0
            }
        }
        
        Ok(())
    }

    /// 从文本中提取常见子字符串
    fn extract_common_substrings(&self, texts: &[String], max_substrings: usize) -> Vec<(Vec<u8>, usize)> {
        let mut substring_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        
        for text in texts {
            let bytes = text.as_bytes();
            // 添加长度最多为4的所有子字符串
            for i in 0..bytes.len() {
                for j in (i + 1)..=(i + 4).min(bytes.len()) {
                    let substring = bytes[i..j].to_vec();
                    *substring_counts.entry(substring).or_insert(0) += 1;
                }
            }
        }
        
        // 按频率排序并返回前max_substrings个
        let mut sorted_substrings: Vec<_> = substring_counts.into_iter().collect();
        sorted_substrings.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_substrings.into_iter().take(max_substrings).collect()
    }

    /// 将字节向量转换为字符串表示
    fn bytes_to_string(&self, bytes: &[u8]) -> String {
        let mut result = String::new();
        for &byte in bytes {
            if byte >= 32 && byte <= 126 {
                // 可打印ASCII字符
                result.push(char::from(byte));
            } else {
                // 非打印字符使用特殊表示
                result.push_str(&format!("<0x{:02X}>", byte));
            }
        }
        result
    }

    /// 使用贪婪算法对字节序列进行分段
    fn segment(&self, bytes: &[u8]) -> Option<Vec<u32>> {
        if bytes.is_empty() {
            return Some(vec![]);
        }
        
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < bytes.len() {
            // 尝试找到最长的匹配
            let mut longest_match = None;
            let mut longest_len = 0;
            
            // 尝试所有可能的标记
            for (token_str, token_id) in &self.base.vocab {
                // 将token字符串转换回字节序列
                let token_bytes = if token_str.starts_with("<0x") && token_str.ends_with(">") {
                    // 特殊字节表示
                    if let Some(hex_str) = token_str.strip_prefix("<0x").and_then(|s| s.strip_suffix(">")) {
                        if let Ok(byte_val) = u8::from_str_radix(hex_str, 16) {
                            vec![byte_val]
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                } else {
                    // 普通字符串
                    token_str.as_bytes().to_vec()
                };
                
                let token_len = token_bytes.len();
                if i + token_len <= bytes.len() && &bytes[i..i + token_len] == &token_bytes {
                    if token_len > longest_len {
                        longest_len = token_len;
                        longest_match = Some(*token_id);
                    }
                }
            }
            
            if let Some(token_id) = longest_match {
                result.push(token_id);
                i += longest_len;
            } else {
                // 如果没有匹配，添加未知标记
                result.push(self.unk_token_id);
                i += 1;
            }
        }
        
        Some(result)
    }
}

impl Tokenizer for WordPieceTokenizer {
    type TokenId = u32;
    
    fn encode(&self, text: &str) -> Result<Vec<Self::TokenId>, String> {
        // 使用基础分词器分割文本
        let parts = self.base.split_text(text)?;
        
        let mut result = Vec::new();
        for part in parts {
            let segment = self.segment(part.as_bytes())
                .ok_or_else(|| "分段失败".to_string())?;
            result.extend(segment);
        }
        
        Ok(result)
    }
    
    fn decode(&self, tokens: &[Self::TokenId]) -> Result<String, String> {
        let mut bytes = Vec::new();
        for &token_id in tokens {
            if let Some(token_str) = self.base.vocab_rev.get(&token_id) {
                // 将token字符串转换回字节序列
                if token_str.starts_with("<0x") && token_str.ends_with(">") {
                    // 特殊字节表示
                    if let Some(hex_str) = token_str.strip_prefix("<0x").and_then(|s| s.strip_suffix(">")) {
                        if let Ok(byte_val) = u8::from_str_radix(hex_str, 16) {
                            bytes.push(byte_val);
                        } else {
                            return Err("无效的字节表示".to_string());
                        }
                    } else {
                        return Err("无效的字节表示".to_string());
                    }
                } else {
                    // 普通字符串
                    bytes.extend(token_str.as_bytes());
                }
            } else {
                return Err(format!("无效的标记ID: {}", token_id));
            }
        }
        
        String::from_utf8(bytes)
            .map_err(|e| format!("UTF-8解码失败: {}", e))
    }
    
    fn train(&mut self, texts: Vec<String>, vocab_size: u32) -> Result<(), String> {
        // 如果请求的词汇表大小小于等于当前词汇表大小，直接返回
        if vocab_size <= self.base.vocab.len() as u32 {
            return Ok(());
        }
        
        // 计算需要提取的子字符串数量
        let current_vocab_size = self.base.vocab.len() as u32;
        let substrings_needed = vocab_size - current_vocab_size;
        
        // 提取常见子字符串
        let common_substrings = self.extract_common_substrings(&texts, substrings_needed as usize);
        
        // 添加常见子字符串到词汇表
        let mut next_id = self.base.vocab.len() as u32; // 从当前词汇表大小开始
        for (substring, _) in common_substrings {
            if next_id >= vocab_size {
                break;
            }
            
            let token_str = self.bytes_to_string(&substring);
            self.base.vocab.insert(token_str.clone(), next_id);
            self.base.vocab_rev.insert(next_id, token_str);
            self.scores.push(0.0); // 初始分数为0
            next_id += 1;
        }
        
        // 如果词汇表还不够大，添加一些随机子字符串
        while next_id < vocab_size {
            // 创建一个随机的1-4字节序列
            let len = (rand::random::<u8>() % 4) + 1;
            let mut substring = Vec::new();
            for _ in 0..len {
                substring.push(rand::random::<u8>());
            }
            
            let token_str = self.bytes_to_string(&substring);
            self.base.vocab.insert(token_str.clone(), next_id);
            self.base.vocab_rev.insert(next_id, token_str);
            self.scores.push(0.0); // 初始分数为0
            next_id += 1;
        }
        
        // 迭代优化词汇表和分数
        // 在实际实现中，这里会执行EM算法优化分数
        // 为简化起见，我们只是设置一些随机分数
        for score in &mut self.scores {
            *score = rand::random::<f64>() * 2.0 - 1.0; // -1.0到1.0之间的随机分数
        }
        
        Ok(())
    }
    
    fn vocab_size(&self) -> usize {
        self.base.vocab_size()
    }
    
    fn save(&self, path: &str) -> Result<(), String> {
        // 使用基础分词器的保存功能
        self.base.save(path)?;
        
        // 保存分数
        let scores_path = format!("{}.scores", path);
        std::fs::write(&scores_path, format!("{}\n", self.unk_token_id))
            .map_err(|e| format!("保存未知标记ID失败: {}", e))?;
        
        for score in &self.scores {
            std::fs::write(&scores_path, format!("{}\n", score))
                .map_err(|e| format!("保存分数失败: {}", e))?;
        }
        
        Ok(())
    }
    
    fn load(&mut self, path: &str) -> Result<(), String> {
        // 使用基础分词器的加载功能
        self.base.load(path)?;
        
        // 加载分数
        let scores_path = format!("{}.scores", path);
        let scores_content = std::fs::read_to_string(&scores_path)
            .map_err(|e| format!("加载分数失败: {}", e))?;
        
        let mut lines = scores_content.lines();
        if let Some(first_line) = lines.next() {
            self.unk_token_id = first_line.parse()
                .map_err(|e| format!("解析未知标记ID失败: {}", e))?;
        }
        
        self.scores.clear();
        for line in lines {
            let score = line.parse()
                .map_err(|e| format!("解析分数失败: {}", e))?;
            self.scores.push(score);
        }
        
        Ok(())
    }
}

impl SubwordTokenizer for WordPieceTokenizer {
    fn get_scores(&self) -> Option<&Vec<f64>> {
        Some(&self.scores)
    }
    
    fn set_scores(&mut self, scores: Vec<f64>) {
        self.scores = scores;
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl WordPieceTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        Self::new_internal()
            .map_err(|e| PyValueError::new_err(e))
    }
    
    #[staticmethod]
    fn with_pattern(pattern: String) -> PyResult<Self> {
        let tokenizer = Self::with_pattern_internal(pattern)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(tokenizer)
    }
    
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        Tokenizer::encode(self, text)
            .map_err(|e| PyValueError::new_err(e))
    }
    
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        Tokenizer::decode(self, &tokens)
            .map_err(|e| PyValueError::new_err(e))
    }
    
    fn train(&mut self, texts: Vec<String>, vocab_size: u32) -> PyResult<()> {
        Tokenizer::train(self, texts, vocab_size)
            .map_err(|e| PyValueError::new_err(e))
    }
    
    fn vocab_size(&self) -> PyResult<usize> {
        Ok(Tokenizer::vocab_size(self))
    }
    
    fn save(&self, path: &str) -> PyResult<()> {
        Tokenizer::save(self, path)
            .map_err(|e| PyValueError::new_err(e))
    }
    
    fn load(&mut self, path: &str) -> PyResult<()> {
        Tokenizer::load(self, path)
            .map_err(|e| PyValueError::new_err(e))
    }
    
    fn get_scores(&self) -> PyResult<Vec<f64>> {
        Ok(self.scores.clone())
    }
    
    fn set_scores(&mut self, scores: Vec<f64>) -> PyResult<()> {
        self.scores = scores;
        Ok(())
    }
}

impl Default for WordPieceTokenizer {
    fn default() -> Self {
        Self::new_internal().unwrap()
    }
}