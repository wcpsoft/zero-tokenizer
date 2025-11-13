use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use crate::base::vocab_manager::VocabManager;
use crate::base::word::Word;

/// 默认的GPT-4风格正则表达式模式，用于分割文本
pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// 分词器基础实现，提供通用功能
#[derive(Clone)]
pub struct TokenizerBase<Id>
where
    Id: Eq + Hash + Clone + std::fmt::Debug,
{
    /// 词汇表管理器（管理 ID <-> String 的双向映射）
    pub vocab: VocabManager<Id, String>,
    /// 正则表达式模式
    pub pattern: String,
    /// 编译后的正则表达式
    pub compiled_pattern: Regex,
}

impl<Id: Clone + Serialize + for<'de> Deserialize<'de> + Eq + Hash + std::fmt::Debug + Default>
    TokenizerBase<Id>
{
    /// 创建新的分词器基础结构
    ///
    /// # Errors
    ///
    /// 当默认正则表达式模式编译失败时返回错误（这种情况极少发生）
    pub fn new() -> Result<Self, String> {
        let pattern = GPT4_PATTERN.to_string();
        let compiled_pattern =
            Regex::new(&pattern).map_err(|e| format!("无效的正则表达式: {}", e))?;

        Ok(Self {
            vocab: VocabManager::new(),
            pattern,
            compiled_pattern,
        })
    }

    /// 使用自定义正则表达式模式创建分词器基础结构
    ///
    /// # Errors
    ///
    /// 当提供的正则表达式模式无效或编译失败时返回错误
    pub fn with_pattern(pattern: String) -> Result<Self, String> {
        let compiled_pattern =
            Regex::new(&pattern).map_err(|e| format!("无效的正则表达式: {}", e))?;

        Ok(Self {
            vocab: VocabManager::new(),
            pattern,
            compiled_pattern,
        })
    }

    /// 添加标记到词汇表
    ///
    /// # Errors
    ///
    /// 当标记已存在于词汇表中或ID已被使用时返回错误
    pub fn add_token(&mut self, token: &str, id: Id) -> Result<(), String> {
        if self.vocab.contains_value(&token.to_string()) {
            return Err(format!("标记 '{}' 已存在于词汇表中", token));
        }

        if self.vocab.contains_id(&id) {
            return Err(format!("ID '{:?}' 已存在于词汇表中", id));
        }

        self.vocab.insert(id, token.to_string());
        Ok(())
    }

    /// 获取标记的ID
    #[must_use]
    pub fn get_token_id(&self, token: &str) -> Option<&Id> {
        self.vocab.get_by_value(&token.to_string())
    }

    /// 获取ID对应的标记
    #[must_use]
    pub fn get_token(&self, id: &Id) -> Option<&String> {
        self.vocab.get_by_id(id)
    }

    /// 获取词汇表大小
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// 使用正则表达式分割文本
    ///
    /// # Errors
    ///
    /// 当正则表达式匹配失败时返回错误。如果正则表达式无法匹配任何内容，
    /// 将使用空格分割作为后备方案
    pub fn split_text(&self, text: &str) -> Result<Vec<String>, String> {
        let parts: Vec<String> = self
            .compiled_pattern
            .find_iter(text)
            .filter_map(|m| m.ok())
            .map(|m| m.as_str().to_string())
            .collect();

        if parts.is_empty() && !text.is_empty() {
            // 如果正则表达式没有匹配任何内容，使用空格分割作为后备
            Ok(text.split_whitespace().map(|s| s.to_string()).collect())
        } else {
            Ok(parts)
        }
    }

    /// 保存分词器到文件
    ///
    /// # Errors
    ///
    /// 当无法创建目录、文件创建失败或写入操作失败时返回错误
    pub fn save(&self, path: &str) -> Result<(), String> {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("创建目录失败: {}", e))?;
        }

        let file = File::create(path).map_err(|e| format!("创建文件失败: {}", e))?;
        let mut writer = io::BufWriter::new(file);

        // 写入正则表达式模式
        writeln!(writer, "pattern: {}", self.pattern)
            .map_err(|e| format!("写入正则表达式失败: {}", e))?;

        // 写入词汇表
        writeln!(writer, "vocab_size: {}", self.vocab.len())
            .map_err(|e| format!("写入词汇表大小失败: {}", e))?;

        for (id, token) in self.vocab.iter() {
            let id_str = serde_json::to_string(id).map_err(|e| format!("序列化ID失败: {}", e))?;
            writeln!(writer, "{} {}", token, id_str)
                .map_err(|e| format!("写入词汇表项失败: {}", e))?;
        }

        Ok(())
    }

    /// 从文件加载分词器
    ///
    /// # Errors
    ///
    /// 当文件不存在、文件格式无效、正则表达式编译失败或ID反序列化失败时返回错误
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("打开文件失败: {}", e))?;
        let reader = BufReader::new(file);

        // 清空当前词汇表
        self.vocab.clear();

        let mut lines = reader.lines();

        // 读取正则表达式模式
        if let Some(Ok(line)) = lines.next() {
            if let Some(pattern_str) = line.strip_prefix("pattern: ") {
                self.pattern = pattern_str.to_string();
                self.compiled_pattern =
                    Regex::new(&self.pattern).map_err(|e| format!("无效的正则表达式: {}", e))?;
            }
        }

        // 跳过词汇表大小行
        let _ = lines.next();

        // 读取词汇表
        for line in lines {
            let line = line.map_err(|e| format!("读取行失败: {}", e))?;
            if line.trim().is_empty() {
                continue;
            }

            let mut parts = line.splitn(2, ' ');
            let token = parts.next().ok_or("无效的词汇表行")?;
            let id_str = parts.next().ok_or("无效的词汇表行")?;

            let id: Id =
                serde_json::from_str(id_str).map_err(|e| format!("反序列化ID失败: {}", e))?;

            self.vocab.insert(id, token.to_string());
        }

        Ok(())
    }

    // 注意：load_vocab_from_dict() 方法已被移除
    // 每个分词器都有自己的实现，因为ID生成策略因分词器而异
    // 参见各分词器模块中的 _load_vocab_from_dict() 方法
}

impl<Id: Clone + Serialize + for<'de> Deserialize<'de> + Eq + Hash + std::fmt::Debug + Default>
    Default for TokenizerBase<Id>
{
    fn default() -> Self {
        Self::new().expect(
            "Default TokenizerBase initialization failed: GPT4_PATTERN should always be valid. \
             This is a programming error - please report this bug.",
        )
    }
}

/// 词对计数映射类型：(Id, Id) -> 计数
pub type PairCounts<Id> = HashMap<(Id, Id), i32>;

/// 词对位置映射类型：(Id, Id) -> 词位置列表
pub type PairPositions<Id> = HashMap<(Id, Id), Vec<usize>>;

/// 并行计算词对频率的通用函数
pub fn count_pairs_parallel<Id: Clone + Eq + Hash + Send + Sync>(
    words: &[Word<Id>],
    counts: &[i32],
) -> (PairCounts<Id>, PairPositions<Id>) {
    use rayon::prelude::*;

    let pair_counts: PairCounts<Id> = words
        .par_iter()
        .enumerate()
        .map(|(i, word)| {
            let mut local_counts = HashMap::new();
            for pair in word.pairs() {
                *local_counts.entry(pair).or_insert(0) += counts[i];
            }
            local_counts
        })
        .reduce(HashMap::new, |mut acc, local_counts| {
            for (pair, count) in local_counts {
                *acc.entry(pair).or_insert(0) += count;
            }
            acc
        });

    let mut where_to_update: PairPositions<Id> = HashMap::new();
    for (i, word) in words.iter().enumerate() {
        for pair in word.pairs() {
            where_to_update.entry(pair).or_default().push(i);
        }
    }

    (pair_counts, where_to_update)
}
