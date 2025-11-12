use std::collections::HashMap as StdHashMap;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

use ahash::{AHashMap, AHashSet};
use dary_heap::OctonaryHeap;
use rayon::prelude::*;

use crate::base::merge_job::MergeJob;
use crate::base::tokenizer_base::{count_pairs_parallel, TokenizerBase};
use crate::base::traits::{MergeBasedTokenizer, Tokenizer};
use crate::base::word::Word;

/// BBPE (字节级BPE) 分词器
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone)]
pub struct BBPETokenizer {
    /// 合并规则
    pub merges: StdHashMap<(u32, u32), u32>,
    /// 词汇表
    pub vocab: StdHashMap<u32, Vec<u8>>,
    /// 反向词汇表
    pub vocab_rev: StdHashMap<Vec<u8>, u32>,
    /// 基础分词器
    pub base: TokenizerBase<u32>,
    /// 基础字符集合（用于初始化词汇表）
    pub base_chars: AHashSet<Vec<u8>>,
    /// 下一个可用的token ID
    pub next_token_id: u32,
}

impl BBPETokenizer {
    /// 创建新的BBPE分词器
    pub fn new_internal() -> Result<Self, String> {
        let base = TokenizerBase::new()?;

        let mut tokenizer = Self {
            merges: StdHashMap::new(),
            vocab: StdHashMap::new(),
            vocab_rev: StdHashMap::new(),
            base,
            base_chars: AHashSet::new(),
            next_token_id: 0,
        };

        // 初始化词汇表，添加所有字节值
        tokenizer.init_vocab();

        Ok(tokenizer)
    }

    /// 使用自定义正则表达式模式创建新的BBPE分词器
    pub fn with_pattern_internal(pattern: String) -> Result<Self, String> {
        let base = TokenizerBase::with_pattern(pattern)?;

        let mut tokenizer = Self {
            merges: StdHashMap::new(),
            vocab: StdHashMap::new(),
            vocab_rev: StdHashMap::new(),
            base,
            base_chars: AHashSet::new(),
            next_token_id: 0,
        };

        // 初始化词汇表，添加所有字节值
        tokenizer.init_vocab();

        Ok(tokenizer)
    }

    /// 从常用汉字字表文件加载基础字符
    pub fn load_base_chars(&mut self, file_path: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::{self, BufRead};

        let file = File::open(file_path)?;
        let reader = io::BufReader::new(file);

        self.base_chars.clear();
        for line in reader.lines() {
            let line = line?;
            let char_str = line.trim();
            if !char_str.is_empty() {
                self.base_chars.insert(char_str.as_bytes().to_vec());
            }
        }

        log::info!("已加载 {} 个基础字符", self.base_chars.len());
        Ok(())
    }

    /// 从dict目录加载初始化词表
    pub fn _load_vocab_from_dict(&mut self, dict_file: &str) -> Result<(), String> {
        use std::fs::File;
        use std::io::{self, BufRead};

        let dict_path = format!("dict/{}", dict_file);
        let file = File::open(&dict_path)
            .map_err(|e| format!("打开词表文件 {} 失败: {}", dict_path, e))?;
        let reader = io::BufReader::new(file);

        // 保留基础字符和字节值，添加新词汇
        let base_vocab_size = self.next_token_id;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("读取行失败: {}", e))?;
            let token = line.trim();
            if token.is_empty() {
                continue;
            }

            // 添加新词汇到词汇表
            let token_bytes = token.as_bytes().to_vec();
            self.vocab.insert(self.next_token_id, token_bytes.clone());
            self.vocab_rev.insert(token_bytes, self.next_token_id);
            self.next_token_id += 1;
        }

        log::info!(
            "已从 {} 加载 {} 个词汇",
            dict_file,
            self.next_token_id - base_vocab_size
        );
        Ok(())
    }

    /// 应用合并规则到ID序列（优化版：贪心合并）
    pub fn apply_merges(&self, ids: &mut Vec<u32>) {
        // 持续应用合并规则，直到没有更多可能的合并
        while ids.len() >= 2 {
            // 在一次扫描中找到所有可以合并的位置
            let mut merges_to_apply = Vec::new();
            let mut i = 0;

            while i < ids.len() - 1 {
                if let Some(&new_id) = self.merges.get(&(ids[i], ids[i + 1])) {
                    merges_to_apply.push((i, new_id));
                    i += 2; // 跳过已合并的pair
                } else {
                    i += 1;
                }
            }

            if merges_to_apply.is_empty() {
                break;
            }

            // 应用所有合并，从后往前以避免索引偏移问题
            let mut new_ids = Vec::with_capacity(ids.len());
            let mut next_merge_idx = 0;
            let mut i = 0;

            while i < ids.len() {
                if next_merge_idx < merges_to_apply.len()
                    && merges_to_apply[next_merge_idx].0 == i
                {
                    // 这个位置需要合并
                    new_ids.push(merges_to_apply[next_merge_idx].1);
                    i += 2; // 跳过被合并的两个token
                    next_merge_idx += 1;
                } else {
                    new_ids.push(ids[i]);
                    i += 1;
                }
            }

            *ids = new_ids;
        }
    }

    /// 给定唯一词的核心增量BPE训练
    fn train_core_incremental(
        &mut self,
        mut words: Vec<Word<u32>>,
        counts: Vec<i32>,
        vocab_size: u32,
    ) {
        let num_merges = vocab_size - self.vocab.len() as u32;
        log::info!("开始增量BBPE训练: 需要计算 {} 次合并", num_merges);
        self.merges.clear();

        // ---- 初始配对计数和更新位置（并行） ----
        log::info!("从 {} 个唯一序列计算初始配对计数", words.len());
        let (pair_counts, where_to_update) = count_pairs_parallel(&words, &counts);

        // ---- 构建堆 ----
        log::info!("使用 {} 个唯一配对构建堆", pair_counts.len());
        let heap = {
            let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
            for (pair, pos) in where_to_update {
                let c = *pair_counts.get(&pair).unwrap_or(&0);
                if c > 0 {
                    let mut merge_job = MergeJob::new(pair, c as u64);
                    merge_job.add_positions(&pos);
                    heap.push(merge_job);
                }
            }
            heap
        };
        let mut heap = heap;

        // ---- 合并循环 ----
        log::info!("开始合并循环");
        let _merges_done = {
            let mut merges_done = 0u32;
            let mut last_log_percent = 0u32;
            let initial_vocab_size = self.vocab.len() as u32;
            let mut pair_counts = pair_counts;

            while merges_done < num_merges {
                let Some(top) = heap.pop() else {
                    break;
                };

                // 如果此配对不再有效（由于之前的合并），跳过它
                if let Some(&current_count) = pair_counts.get(&top.pair) {
                    if current_count as u64 != top.count {
                        continue;
                    }
                } else {
                    continue;
                }

                let new_id = initial_vocab_size + merges_done;
                self.merges.insert(top.pair, new_id);

                // 创建新标记
                let new_token_bytes = {
                    let mut new_token_bytes = self.vocab[&top.pair.0].clone();
                    new_token_bytes.extend(&self.vocab[&top.pair.1]);
                    new_token_bytes
                };
                self.vocab.insert(new_id, new_token_bytes.clone());
                self.vocab_rev.insert(new_token_bytes, new_id);

                // 更新受影响的词
                let (updated_pairs, updated_where) = {
                    let mut updated_pairs: AHashMap<(u32, u32), i32> = AHashMap::new();
                    let mut updated_where: AHashMap<(u32, u32), AHashSet<usize>> = AHashMap::new();

                    for &word_idx in &top.pos {
                        let deltas =
                            words[word_idx].merge_pair(top.pair.clone(), new_id, |a, b| a == b);
                        for (pair, delta) in deltas {
                            *updated_pairs.entry(pair).or_insert(0) += delta * counts[word_idx];
                            updated_where.entry(pair).or_default().insert(word_idx);
                        }
                    }
                    (updated_pairs, updated_where)
                };

                // 更新全局计数
                for (pair, delta) in updated_pairs {
                    let entry = pair_counts.entry(pair).or_insert(0);
                    *entry += delta;

                    if *entry <= 0 {
                        pair_counts.remove(&pair);
                    } else if let Some(pos_set) = updated_where.get(&pair) {
                        let mut merge_job = MergeJob::new(pair, *entry as u64);
                        merge_job.add_positions(&pos_set.iter().cloned().collect::<Vec<_>>());
                        heap.push(merge_job);
                    }
                }

                merges_done += 1;

                // 每10%记录一次进度
                let percent = merges_done * 100 / num_merges;
                if percent > last_log_percent {
                    log::info!("训练进度: {}% ({} 次合并)", percent, merges_done);
                    last_log_percent = percent;
                }
            }
            merges_done
        };
    }

    /// 初始化词汇表
    fn init_vocab(&mut self) {
        log::info!("初始化词汇表");
        self.vocab.clear();
        self.vocab_rev.clear();

        // 首先添加基础字符（如果有）
        for (i, char_bytes) in self.base_chars.iter().enumerate() {
            self.vocab.insert(i as u32, char_bytes.clone());
            self.vocab_rev.insert(char_bytes.clone(), i as u32);
        }

        // 然后添加所有字节值（从0到255）
        let mut next_id = self.base_chars.len() as u32;
        for byte in 0..=255u8 {
            let byte_vec = vec![byte];
            if !self.vocab_rev.contains_key(&byte_vec) {
                self.vocab.insert(next_id, byte_vec.clone());
                self.vocab_rev.insert(byte_vec, next_id);
                next_id += 1;
            }
        }
        self.next_token_id = next_id;
        log::info!(
            "已初始化词汇表，包含 {} 个基础字符和 {} 个字节值",
            self.base_chars.len(),
            next_id - self.base_chars.len() as u32
        );
    }
}

impl Default for BBPETokenizer {
    fn default() -> Self {
        Self::new_internal().unwrap()
    }
}

/// 公共方法，将暴露给Python的BBPETokenizer类。
#[cfg(feature = "python")]
#[pymethods]
impl BBPETokenizer {
    /// 创建一个新的BBPE分词器，使用默认的GPT-4风格正则表达式模式
    #[new]
    pub fn new() -> PyResult<Self> {
        Self::new_internal()
            .map_err(|e| crate::error::TokenizerError::InitializationError { message: e }.into())
    }

    /// 使用自定义正则表达式模式创建新的BBPE分词器
    #[staticmethod]
    pub fn with_pattern(pattern: String) -> PyResult<Self> {
        Self::with_pattern_internal(pattern)
            .map_err(|e| crate::error::TokenizerError::InitializationError { message: e }.into())
    }

    /// 从常用汉字字表文件加载基础字符
    #[cfg(feature = "python")]
    #[pyo3(name = "load_base_chars")]
    pub fn py_load_base_chars(&mut self, file_path: String) -> PyResult<()> {
        self.load_base_chars(&file_path)?;

        // 重新初始化词汇表以包含新加载的基础字符
        self.init_vocab();

        Ok(())
    }

    /// 从dict目录加载初始化词表
    #[cfg(feature = "python")]
    #[pyo3(name = "load_vocab_from_dict")]
    pub fn py_load_vocab_from_dict(&mut self, dict_file: String) -> PyResult<()> {
        self._load_vocab_from_dict(&dict_file)
            .map_err(|e| crate::error::TokenizerError::LoadError { message: e }.into())
    }

    /// 从Python迭代器训练分词器
    #[cfg(feature = "python")]
    #[pyo3(name = "train_from_iterator")]
    pub fn py_train_from_iterator(
        &mut self,
        texts: Vec<String>,
        vocab_size: usize,
        _show_progress: bool,
    ) -> PyResult<()> {
        self.train(texts, vocab_size as u32)
            .map_err(|e| crate::error::TokenizerError::TrainingError { message: e }.into())
    }

    /// 从迭代器训练分词器
    #[cfg(feature = "python")]
    #[pyo3(name = "train")]
    pub fn py_train(&mut self, texts: Vec<String>, vocab_size: usize) -> PyResult<()> {
        self.train(texts, vocab_size as u32)
            .map_err(|e| crate::error::TokenizerError::TrainingError { message: e }.into())
    }

    /// 从迭代器训练分词器
    #[cfg(feature = "python")]
    #[pyo3(name = "train_from_iterator_stream")]
    pub fn train_from_iterator(
        &mut self,
        texts: Vec<String>,
        vocab_size: usize,
        _show_progress: bool,
    ) -> PyResult<()> {
        self.train(texts, vocab_size as u32)
            .map_err(|e| crate::error::TokenizerError::TrainingError { message: e }.into())
    }

    /// 返回正则表达式模式
    #[cfg(feature = "python")]
    #[getter]
    pub fn get_pattern(&self) -> String {
        self.base.pattern.clone()
    }

    /// 获取合并等级映射
    #[cfg(feature = "python")]
    #[pyo3(name = "get_mergeable_ranks")]
    pub fn py_get_mergeable_ranks(&self) -> StdHashMap<(u32, u32), u32> {
        self.get_mergeable_ranks()
    }

    /// 将文本编码为token IDs
    #[cfg(feature = "python")]
    #[pyo3(name = "encode")]
    pub fn py_encode(&self, text: &str) -> PyResult<Vec<u32>> {
        self.encode(text)
            .map_err(|e| crate::error::TokenizerError::EncodingError { message: e }.into())
    }

    /// 将token IDs解码为文本
    #[cfg(feature = "python")]
    #[pyo3(name = "decode")]
    pub fn py_decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.decode(&tokens)
            .map_err(|e| crate::error::TokenizerError::DecodingError { message: e }.into())
    }

    /// 批量编码文本为token IDs（并行处理）
    #[cfg(feature = "python")]
    #[pyo3(name = "encode_batch")]
    pub fn py_encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        // 使用rayon并行处理所有文本
        let results: Result<Vec<Vec<u32>>, String> = texts
            .par_iter()
            .map(|text| self.encode(text))
            .collect();

        results.map_err(|e| {
            crate::error::TokenizerError::EncodingError { message: e }.into()
        })
    }

    /// 批量解码token IDs为文本（并行处理）
    #[cfg(feature = "python")]
    #[pyo3(name = "decode_batch")]
    pub fn py_decode_batch(&self, token_lists: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        // 使用rayon并行处理所有token列表
        let results: Result<Vec<String>, String> = token_lists
            .par_iter()
            .map(|tokens| self.decode(tokens))
            .collect();

        results.map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// 获取词汇表大小
    #[cfg(feature = "python")]
    #[pyo3(name = "vocab_size")]
    pub fn py_vocab_size(&self) -> usize {
        self.vocab_size()
    }

    /// 获取词汇表
    #[cfg(feature = "python")]
    #[pyo3(name = "get_vocab")]
    pub fn py_get_vocab(&self) -> StdHashMap<u32, Vec<u8>> {
        self.vocab.clone()
    }

    /// 获取反向词汇表
    #[cfg(feature = "python")]
    #[pyo3(name = "get_vocab_rev")]
    pub fn py_get_vocab_rev(&self) -> StdHashMap<Vec<u8>, u32> {
        self.vocab_rev.clone()
    }

    /// 获取合并规则
    #[cfg(feature = "python")]
    #[pyo3(name = "get_merges")]
    pub fn py_get_merges(&self) -> StdHashMap<(u32, u32), u32> {
        self.merges.clone()
    }

    /// 保存分词器到文件
    #[cfg(feature = "python")]
    #[pyo3(name = "save")]
    pub fn py_save(&self, path: String) -> PyResult<()> {
        self.save(&path)
            .map_err(|e| crate::error::TokenizerError::ModelSaveError { message: e }.into())
    }

    /// 从文件加载分词器
    #[cfg(feature = "python")]
    #[pyo3(name = "load")]
    pub fn py_load(&mut self, path: String) -> PyResult<()> {
        self.load(&path)
            .map_err(|e| crate::error::TokenizerError::ModelLoadError { message: e }.into())
    }
}

impl BBPETokenizer {
    /// 获取合并等级映射
    pub fn get_mergeable_ranks(&self) -> StdHashMap<(u32, u32), u32> {
        self.merges.clone()
    }
}

impl Tokenizer for BBPETokenizer {
    type TokenId = u32;

    fn encode(&self, text: &str) -> Result<Vec<Self::TokenId>, String> {
        // 使用正则表达式分割文本
        let parts = self.base.split_text(text)?;

        let mut result = Vec::new();

        for part in parts {
            if part.is_empty() {
                continue;
            }

            // 将每个部分转换为字节ID
            let mut ids: Vec<u32> = Vec::new();
            for &byte in part.as_bytes() {
                let byte_vec = vec![byte];
                if let Some(&id) = self.vocab_rev.get(&byte_vec) {
                    ids.push(id);
                } else {
                    // 这种情况不应该发生，因为我们已经初始化了所有可能的字节
                    return Err(format!("未找到字节 {} 对应的ID", byte));
                }
            }

            // 应用合并规则
            self.apply_merges(&mut ids);
            result.extend(ids);
        }

        // 如果没有匹配到任何内容，退回到简单分割
        if result.is_empty() {
            for word in text.split_whitespace() {
                let mut ids: Vec<u32> = Vec::new();
                for &byte in word.as_bytes() {
                    let byte_vec = vec![byte];
                    if let Some(&id) = self.vocab_rev.get(&byte_vec) {
                        ids.push(id);
                    } else {
                        // 这种情况不应该发生，因为我们已经初始化了所有可能的字节
                        return Err(format!("未找到字节 {} 对应的ID", byte));
                    }
                }

                // 应用合并规则
                self.apply_merges(&mut ids);
                result.extend(ids);
            }
        }

        Ok(result)
    }

    fn decode(&self, tokens: &[Self::TokenId]) -> Result<String, String> {
        let mut bytes = Vec::new();

        for &id in tokens {
            if let Some(token_bytes) = self.vocab.get(&id) {
                bytes.extend_from_slice(token_bytes);
            } else {
                return Err(format!("未找到ID {} 对应的词汇", id));
            }
        }

        match String::from_utf8(bytes) {
            Ok(s) => Ok(s),
            Err(e) => Err(format!("UTF-8解码失败: {}", e)),
        }
    }

    fn train(&mut self, texts: Vec<String>, vocab_size: u32) -> Result<(), String> {
        log::info!("开始BBPE训练，目标词汇表大小: {}", vocab_size);

        // 验证词汇表大小
        if vocab_size < 256 {
            return Err("词汇表大小必须至少为256".to_string());
        }

        // 只有在词汇表为空时才初始化
        if self.vocab.is_empty() {
            self.init_vocab();
        }

        // 将文本转换为词序列
        log::info!("处理 {} 个文本样本", texts.len());
        let (words, counts) = {
            let mut words = Vec::new();
            let mut counts = Vec::new();

            for text in &texts {
                // 使用正则表达式分割文本
                let parts = self.base.split_text(text)?;

                for part in parts {
                    if part.is_empty() {
                        continue;
                    }

                    // 将词转换为字节ID - 通过vocab_rev查找每个字节对应的ID
                    let ids: Vec<u32> = part
                        .bytes()
                        .map(|b| {
                            let byte_vec = vec![b];
                            *self
                                .vocab_rev
                                .get(&byte_vec)
                                .unwrap_or_else(|| panic!("字节 {} 在vocab_rev中不存在", b))
                        })
                        .collect();
                    words.push(Word::new(ids));
                    counts.push(1);
                }

                // 如果没有匹配到任何内容，退回到简单分割
                if words.is_empty() {
                    log::warn!("正则表达式未匹配，使用简单分割");
                    for word in text.split_whitespace() {
                        let ids: Vec<u32> = word
                            .bytes()
                            .map(|b| {
                                let byte_vec = vec![b];
                                *self
                                    .vocab_rev
                                    .get(&byte_vec)
                                    .unwrap_or_else(|| panic!("字节 {} 在vocab_rev中不存在", b))
                            })
                            .collect();
                        words.push(Word::new(ids));
                        counts.push(1);
                    }
                }
            }
            (words, counts)
        };

        // 使用增量训练核心
        self.train_core_incremental(words, counts, vocab_size);
        log::info!("BBPE训练完成，最终词汇表大小: {}", self.vocab.len());

        Ok(())
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn save(&self, path: &str) -> Result<(), String> {
        // 使用基础分词器的保存方法
        self.base.save(path)?;

        // 保存BBPE特定的数据
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("打开文件失败: {}", e))?;

        // 保存基础字符
        writeln!(file, "base_chars: {}", self.base_chars.len())
            .map_err(|e| format!("写入基础字符数量失败: {}", e))?;

        for char_bytes in &self.base_chars {
            let char_str = String::from_utf8_lossy(char_bytes);
            writeln!(file, "base_char: {}", char_str)
                .map_err(|e| format!("写入基础字符失败: {}", e))?;
        }

        // 保存词汇表
        writeln!(file, "vocab: {}", self.vocab.len())
            .map_err(|e| format!("写入词汇表数量失败: {}", e))?;

        for (id, bytes) in &self.vocab {
            let byte_str = bytes
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(file, "vocab_entry: {} {}", id, byte_str)
                .map_err(|e| format!("写入词汇表条目失败: {}", e))?;
        }

        // 保存合并规则
        writeln!(file, "merges: {}", self.merges.len())
            .map_err(|e| format!("写入合并规则数量失败: {}", e))?;

        for ((a, b), &rank) in &self.merges {
            writeln!(file, "merge: {} {} {}", a, b, rank)
                .map_err(|e| format!("写入合并规则失败: {}", e))?;
        }

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<(), String> {
        // 使用基础分词器的加载方法
        self.base.load(path)?;

        // 加载BBPE特定的数据
        use std::fs::File;
        use std::io::{self, BufRead};

        let file = File::open(path).map_err(|e| format!("打开文件失败: {}", e))?;
        let reader = io::BufReader::new(file);

        let lines = reader.lines();
        let mut in_base_chars = false;
        let mut in_vocab = false;
        let mut in_merges = false;

        // 清空当前数据
        self.base_chars.clear();
        self.vocab.clear();
        self.vocab_rev.clear();
        self.merges.clear();

        for line in lines {
            let line = line.map_err(|e| format!("读取行失败: {}", e))?;
            let line = line.trim();

            if line.starts_with("base_chars: ") {
                in_base_chars = true;
                in_vocab = false;
                in_merges = false;
                continue;
            } else if line.starts_with("vocab: ") {
                in_base_chars = false;
                in_vocab = true;
                in_merges = false;
                continue;
            } else if line.starts_with("merges: ") {
                in_base_chars = false;
                in_vocab = false;
                in_merges = true;
                continue;
            } else if line.starts_with("base_char: ") {
                if in_base_chars {
                    let char_str = line[11..].to_string();
                    self.base_chars.insert(char_str.as_bytes().to_vec());
                }
            } else if line.starts_with("vocab_entry: ") {
                if in_vocab {
                    let parts: Vec<&str> = line[12..].split_whitespace().collect();
                    if parts.len() >= 2 {
                        let id = parts[0]
                            .parse::<u32>()
                            .map_err(|e| format!("解析词汇表ID失败: {}", e))?;
                        let bytes: Result<Vec<u8>, _> =
                            parts[1..].iter().map(|s| s.parse::<u8>()).collect();
                        let bytes = bytes.map_err(|e| format!("解析字节失败: {}", e))?;

                        self.vocab.insert(id, bytes.clone());
                        self.vocab_rev.insert(bytes, id);
                    }
                }
            } else if line.starts_with("merge: ") {
                if in_merges {
                    let parts: Vec<&str> = line[6..].split_whitespace().collect();
                    if parts.len() == 3 {
                        let a = parts[0]
                            .parse::<u32>()
                            .map_err(|e| format!("解析合并规则失败: {}", e))?;
                        let b = parts[1]
                            .parse::<u32>()
                            .map_err(|e| format!("解析合并规则失败: {}", e))?;
                        let rank = parts[2]
                            .parse::<u32>()
                            .map_err(|e| format!("解析合并规则失败: {}", e))?;
                        self.merges.insert((a, b), rank);
                    }
                }
            }
        }

        Ok(())
    }
}

impl MergeBasedTokenizer for BBPETokenizer {
    fn apply_merges(&mut self, tokens: &mut Vec<Self::TokenId>) -> Result<(), String> {
        // 应用合并规则，直到没有更多合并可以应用
        let mut changed = true;
        while changed {
            changed = false;
            let mut new_tokens = Vec::new();
            let mut i = 0;

            while i < tokens.len() {
                if i + 1 < tokens.len() {
                    // 检查当前token对是否可以合并
                    if let Some(&new_id) = self.merges.get(&(tokens[i], tokens[i + 1])) {
                        new_tokens.push(new_id);
                        i += 2;
                        changed = true;
                    } else {
                        new_tokens.push(tokens[i]);
                        i += 1;
                    }
                } else {
                    new_tokens.push(tokens[i]);
                    i += 1;
                }
            }

            *tokens = new_tokens;
        }

        Ok(())
    }

    fn get_merges(&self) -> &StdHashMap<(Self::TokenId, Self::TokenId), Self::TokenId> {
        &self.merges
    }

    fn set_merges(&mut self, merges: StdHashMap<(Self::TokenId, Self::TokenId), Self::TokenId>) {
        self.merges = merges;
    }
}
