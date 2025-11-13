#[cfg(feature = "python")]
use std::collections::HashMap as StdHashMap;

#[cfg(feature = "python")]
use dary_heap::OctonaryHeap;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use ahash::{AHashMap, AHashSet};
#[cfg(feature = "python")]
use compact_str::CompactString;
#[cfg(feature = "python")]
use rayon::prelude::*;

#[cfg(feature = "python")]
use crate::base::merge_job::MergeJob;
#[cfg(feature = "python")]
use crate::base::tokenizer_base::{count_pairs_parallel, TokenizerBase, GPT4_PATTERN};
#[cfg(feature = "python")]
use crate::base::traits::{MergeBasedTokenizer, Tokenizer as TokenizerTrait};
#[cfg(feature = "python")]
use crate::base::vocab_manager::VocabManager;
#[cfg(feature = "python")]
use crate::base::word::Word;

/// 词ID类型
pub type WordId = u32;

/// BPE分词器实现，参考template.rs并结合src/base基础组件
#[cfg(feature = "python")]
#[pyclass]
pub struct Tokenizer {
    /// 合并规则：(token_a, token_b) -> new_token_id
    #[pyo3(get, set)]
    pub merges: StdHashMap<(WordId, WordId), WordId>,
    /// 基础分词器，用于文本分割和基础功能
    pub base: TokenizerBase<u32>,
    /// 词汇表管理器（管理 ID <-> String 的双向映射）
    pub vocab: VocabManager<WordId, String>,
    /// 下一个可用的token ID
    pub next_token_id: WordId,
}

#[cfg(feature = "python")]
impl Tokenizer {
    /// 创建新的分词器
    pub fn _new_internal() -> Result<Self, String> {
        let base = TokenizerBase::new()?;

        let tokenizer = Self {
            merges: StdHashMap::new(),
            base,
            vocab: VocabManager::new(),
            next_token_id: 0, // 从0开始，训练时动态分配
        };

        // vocab将在训练时按需初始化，无需预先分配所有Unicode字符
        Ok(tokenizer)
    }

    /// 使用自定义正则表达式模式创建新的分词器
    pub fn _with_pattern_internal(pattern: String) -> Result<Self, String> {
        let base = TokenizerBase::with_pattern(pattern)?;

        let tokenizer = Self {
            merges: StdHashMap::new(),
            base,
            vocab: VocabManager::new(),
            next_token_id: 0, // 从0开始，训练时动态分配
        };

        // vocab将在训练时按需初始化，无需预先分配所有Unicode字符
        Ok(tokenizer)
    }

    /// 从常用汉字字表文件加载基础字符
    pub fn _load_base_chars(&mut self, file_path: &str) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::{self, BufRead};

        let file = File::open(file_path)?;
        let reader = io::BufReader::new(file);

        // 清除现有词汇表中256以上的条目
        let ids_to_remove: Vec<WordId> =
            self.vocab.ids().filter(|&&id| id >= 256).copied().collect();

        for id in ids_to_remove {
            self.vocab.remove_by_id(&id);
        }
        self.next_token_id = 256;

        for line in reader.lines() {
            let line = line?;
            let char_str = line.trim();
            if !char_str.is_empty() {
                self.vocab.insert(self.next_token_id, char_str.to_string());
                self.next_token_id += 1;
            }
        }

        log::info!("已加载 {} 个基础字符", self.vocab.len() - 256);
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

        // 清除现有词汇表中256以上的条目
        let ids_to_remove: Vec<WordId> =
            self.vocab.ids().filter(|&&id| id >= 256).copied().collect();

        for id in ids_to_remove {
            self.vocab.remove_by_id(&id);
        }
        self.next_token_id = 256;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("读取行失败: {}", e))?;
            let token = line.trim();
            if token.is_empty() {
                continue;
            }

            // 添加新词汇到词汇表
            self.vocab.insert(self.next_token_id, token.to_string());
            self.next_token_id += 1;
        }

        log::info!("已从 {} 加载 {} 个词汇", dict_file, self.vocab.len() - 256);
        Ok(())
    }

    /// 获取合并等级映射
    pub fn _get_mergeable_ranks_internal(&self) -> StdHashMap<(WordId, WordId), u32> {
        let mut ranks = StdHashMap::new();

        // 首先添加基础字符的映射（0-255）
        for i in 0..256u32 {
            ranks.insert((i, 0), i);
        }

        // 然后添加合并规则的映射
        for (pair, &new_id) in &self.merges {
            ranks.insert(pair.clone(), new_id);
        }

        ranks
    }

    /// 应用合并规则到标记序列
    pub fn _apply_merges(&mut self, tokens: &mut Vec<u32>) -> Result<(), String> {
        // 创建Word并应用合并规则
        let mut word = Word::new(tokens.clone());

        // 持续应用合并规则，直到没有更多可能的合并
        let mut changed = true;
        while changed {
            changed = false;

            // 查找可以合并的词对
            for (pair, &new_id) in &self.merges {
                let mut i = 0;
                while i < word.ids().len() - 1 {
                    if word.ids()[i] == pair.0 && word.ids()[i + 1] == pair.1 {
                        // 执行合并
                        word.merge_pair(pair.clone(), new_id, |a, b| a == b);
                        changed = true;
                        break;
                    }
                    i += 1;
                }

                if changed {
                    break;
                }
            }
        }

        // 更新tokens为合并后的结果
        *tokens = word.ids().to_vec();

        Ok(())
    }

    /// 给定唯一词的核心增量BPE训练
    fn _train_core_incremental(&mut self, mut words: Vec<Word<WordId>>, vocab_size: u32) {
        // 确保词汇表大小不小于基础字符数量
        let vocab_size = vocab_size.max(0x110000);
        let num_merges = vocab_size - 0x110000;
        log::info!("开始增量BPE训练: 需要计算 {} 次合并", num_merges);
        self.merges.clear();

        // ---- 初始配对计数和更新位置（并行） ----
        log::info!("从 {} 个唯一序列计算初始配对计数", words.len());
        let counts: Vec<i32> = vec![1; words.len()]; // 每个词的初始计数为1
        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);

        // ---- 构建堆 ----
        log::info!("使用 {} 个唯一配对构建堆", pair_counts.len());
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let c = *pair_counts.get(&pair).unwrap_or(&0);
            if c > 0 {
                let mut merge_job = MergeJob::new(pair, c as u64);
                merge_job.add_positions(&pos);
                heap.push(merge_job);
            }
        }

        // ---- 合并循环 ----
        log::info!("开始合并循环");
        let mut merges_done = 0u32;
        let mut last_log_percent = 0u32;

        while merges_done < num_merges {
            let Some(top) = heap.pop() else {
                // 如果没有更多的配对可以合并，停止训练
                log::info!(
                    "没有更多配对可合并，停止训练。已完成 {} 次合并，词汇表大小: {}",
                    merges_done,
                    self.vocab.len()
                );
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

            // 执行合并
            let new_id = self.next_token_id;
            self.next_token_id += 1;
            self.merges.insert(top.pair.clone(), new_id);

            // 更新词汇表
            if let (Some(a_text), Some(b_text)) = (
                self.vocab.get_by_id(&top.pair.0),
                self.vocab.get_by_id(&top.pair.1),
            ) {
                // 直接合并文本，不进行字节转换
                let merged_text = format!("{}{}", a_text, b_text);
                self.vocab.insert(new_id, merged_text.clone());
            }

            // 更新受影响的词
            let mut updated_pairs: AHashMap<(WordId, WordId), i32> = AHashMap::new();
            let mut updated_where: AHashMap<(WordId, WordId), AHashSet<usize>> = AHashMap::new();

            for &word_idx in &top.pos {
                let deltas = words[word_idx].merge_pair(top.pair.clone(), new_id, |a, b| a == b);
                for (pair, delta) in deltas {
                    *updated_pairs.entry(pair.clone()).or_insert(0) += delta;
                    updated_where.entry(pair).or_default().insert(word_idx);
                }
            }

            // 更新全局计数
            for (pair, delta) in updated_pairs {
                let entry = pair_counts.entry(pair.clone()).or_insert(0);
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

        log::info!(
            "训练完成，词汇表大小: {}, next_token_id: {}",
            self.vocab.len(),
            self.next_token_id
        );
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Tokenizer {
    /// 创建新的BPE分词器
    #[new]
    pub fn new() -> PyResult<Self> {
        Self::_new_internal().map_err(|e| PyValueError::new_err(e))
    }

    /// 使用自定义正则表达式模式创建新的BPE分词器
    #[staticmethod]
    pub fn with_pattern(pattern: String) -> PyResult<Self> {
        let mut tokenizer = Self::_new_internal().map_err(|e| PyValueError::new_err(e))?;
        tokenizer.base.pattern = pattern.clone();
        tokenizer.base.compiled_pattern = fancy_regex::Regex::new(&pattern)
            .map_err(|e| PyValueError::new_err(format!("无效的正则表达式: {}", e)))?;
        Ok(tokenizer)
    }

    /// 编码文本为token IDs
    pub fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        // 实际的编码实现
        self._encode_internal(text).map_err(|e| {
            crate::error::TokenizerError::EncodingError {
                message: e.to_string(),
            }
            .into()
        })
    }

    /// 解码token IDs为文本
    pub fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        // 实际的解码实现
        self.decode_internal(tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// 批量编码文本为token IDs（并行处理）
    pub fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        use rayon::prelude::*;

        // 使用rayon并行处理所有文本
        let results: Result<Vec<Vec<u32>>, _> = texts
            .par_iter()
            .map(|text| {
                self._encode_internal(text).map_err(|e| {
                    crate::error::TokenizerError::EncodingError {
                        message: e.to_string(),
                    }
                })
            })
            .collect();

        results.map_err(|e| e.into())
    }

    /// 批量解码token IDs为文本（并行处理）
    pub fn decode_batch(&self, token_lists: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        use rayon::prelude::*;

        // 使用rayon并行处理所有token列表
        let results: Result<Vec<String>, _> = token_lists
            .par_iter()
            .map(|tokens| self.decode_internal(tokens.clone()))
            .collect();

        results.map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// 训练分词器
    pub fn train(&mut self, texts: Vec<String>, vocab_size: u32) -> PyResult<()> {
        TokenizerTrait::train(self, texts, vocab_size).map_err(|e| PyValueError::new_err(e))
    }

    /// 获取词汇表大小
    pub fn get_vocab_size(&self) -> usize {
        self._vocab_size()
    }

    /// 获取词汇表
    pub fn get_vocab(&self) -> std::collections::HashMap<u32, String> {
        // 转换词汇表类型
        self.vocab.iter().map(|(&k, v)| (k, v.clone())).collect()
    }

    /// 获取正则表达式模式
    pub fn get_pattern(&self) -> String {
        self.base.pattern.clone()
    }

    /// 获取合并等级映射
    pub fn get_mergeable_ranks(&self) -> std::collections::HashMap<(u32, u32), u32> {
        // 转换合并等级映射类型
        self._get_mergeable_ranks_internal()
            .into_iter()
            .map(|((a, b), c)| ((a, b), c))
            .collect()
    }

    /// 保存分词器
    pub fn save(&self, path: &str) -> PyResult<()> {
        TokenizerTrait::save(self, path)
            .map_err(|e| crate::error::TokenizerError::ModelSaveError { message: e }.into())
    }

    /// 加载分词器
    pub fn load(&mut self, path: &str) -> PyResult<()> {
        TokenizerTrait::load(self, path).map_err(|e| PyValueError::new_err(e))
    }

    /// 从常用汉字字表文件加载基础字符
    pub fn load_base_chars(&mut self, file_path: &str) -> PyResult<()> {
        self._load_base_chars(file_path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// 从常用汉字字表文件加载基础字符
    #[pyo3(name = "load_base_chars_bpe")]
    pub fn py_load_base_chars(&mut self, file_path: String) -> PyResult<()> {
        self._load_base_chars(&file_path)
            .map_err(|e| crate::error::TokenizerError::IoError { source: e }.into())
    }

    /// 从dict目录加载初始化词表
    #[pyo3(name = "load_vocab_from_dict")]
    pub fn py_load_vocab_from_dict(&mut self, dict_file: String) -> PyResult<()> {
        self._load_vocab_from_dict(&dict_file)
            .map_err(|e| crate::error::TokenizerError::LoadError { message: e }.into())
    }

    /// 从Python迭代器训练分词器
    #[cfg(feature = "python")]
    #[pyo3(name = "train_from_iterator")]
    pub fn py_train_from_iterator(&mut self, texts: Vec<String>, vocab_size: u32) -> PyResult<()> {
        self.train(texts, vocab_size)
            .map_err(|e| PyValueError::new_err(format!("训练失败: {}", e)))
    }

    /// 从流式迭代器训练（并行摄取）
    #[cfg(feature = "python")]
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None))]
    #[pyo3(text_signature = "(self, iterator, vocab_size, buffer_size=8192, pattern=None)")]
    #[pyo3(name = "train_from_iterator_stream")]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
    ) -> PyResult<()> {
        // 使用提供的模式或默认为GPT-4模式
        let pattern_str = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());

        // 更新存储的模式并编译它
        self.base.pattern = pattern_str.clone();
        self.base.compiled_pattern = fancy_regex::Regex::new(&pattern_str).map_err(|e| {
            crate::error::TokenizerError::InvalidRegex {
                message: e.to_string(),
            }
        })?;

        // 准备一个真正的Python迭代器对象
        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Bound::from_borrowed_ptr_or_err(
                py,
                pyo3::ffi::PyObject_GetIter(iterator.as_ptr()),
            )
            .map_err(|e| crate::error::TokenizerError::InvalidIterator {
                message: e.to_string(),
            })?
            .into()
        };

        // 全局块计数
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();

        // 临时缓冲区，我们在GIL下填充它
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!(
            "Processing sequences from iterator (buffer_size: {})",
            buffer_size
        );
        let mut total_sequences = 0u64;

        // 辅助函数：在`buf`中填充最多`buffer_size`个字符串来自Python迭代器
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::with_gil(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    // next(it)
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract().map_err(|e| {
                                crate::error::TokenizerError::InvalidInput {
                                    message: e.to_string(),
                                }
                            })?;
                            buf.push(s);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py).into());
                            } else {
                                return Ok(true); // exhausted
                            }
                        }
                    }
                }
            })
        };

        // 流摄取循环：在GIL下填充，在GIL外处理（并行）
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_sequences += buf.len() as u64;

            let pattern = self.base.compiled_pattern.clone();
            let local: AHashMap<CompactString, i32> = py.allow_threads(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                        for mat in pattern.find_iter(s) {
                            let piece = match mat {
                                Ok(m) => m.as_str(),
                                Err(_) => continue,
                            };
                            *m.entry(CompactString::from(piece)).or_default() += 1;
                        }
                        m
                    })
                    .reduce(
                        || AHashMap::new(),
                        |mut a, b| {
                            for (k, v) in b {
                                *a.entry(k).or_default() += v;
                            }
                            a
                        },
                    )
            });

            // 合并局部到全局（单线程）
            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }

            if exhausted {
                break;
            }
        }
        log::info!(
            "Processed {} sequences total, {} unique",
            total_sequences,
            counts.len()
        );

        // 物化词和计数
        let mut words = Vec::with_capacity(counts.len());
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, c) in counts.into_iter() {
            // 将文本分割为字符序列
            let mut ids = Vec::new();
            for ch in chunk.chars() {
                // 直接使用字符的Unicode码点作为token ID
                let code_point = ch as u32;
                // 确保字符在词汇表中
                if !self.vocab.contains_id(&code_point) {
                    self.vocab.insert(code_point, ch.to_string());
                    if code_point >= self.next_token_id {
                        self.next_token_id = code_point + 1;
                    }
                }
                ids.push(code_point);
            }

            words.push(Word::new(ids));
            cvec.push(c);
        }

        self._train_core_incremental(words, vocab_size);
        Ok(())
    }

    /// 返回正则表达式模式
    #[cfg(feature = "python")]
    #[pyo3(text_signature = "(self)")]
    pub fn _get_pattern(&self) -> String {
        self.base.pattern.clone()
    }

    /// 获取合并等级映射
    #[cfg(feature = "python")]
    #[pyo3(name = "_get_mergeable_ranks")]
    pub fn py_get_mergeable_ranks(&self) -> StdHashMap<(WordId, WordId), u32> {
        self._get_mergeable_ranks_internal()
    }

    /// 获取词汇表大小
    #[cfg(feature = "python")]
    pub fn _vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// 获取词汇表
    #[cfg(feature = "python")]
    pub fn _get_vocab(&self) -> StdHashMap<WordId, String> {
        self.vocab.id_map().clone()
    }

    /// 将文本编码为token IDs
    #[cfg(feature = "python")]
    #[pyo3(text_signature = "(self, text)")]
    pub fn _py_encode(&self, text: &str) -> Vec<u32> {
        match self._encode_internal(text) {
            Ok(tokens) => tokens,
            Err(e) => {
                log::error!("编码失败: {}", e);
                Vec::new()
            }
        }
    }

    /// 将token IDs解码为文本
    #[cfg(feature = "python")]
    #[pyo3(text_signature = "(self, tokens)")]
    pub fn _py_decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.decode_internal(tokens).map_err(|e| {
            crate::error::TokenizerError::DecodingError {
                message: e.to_string(),
            }
            .into()
        })
    }

    /// 内部编码实现
    fn _encode_internal(&self, text: &str) -> Result<Vec<u32>, crate::error::TokenizerError> {
        // 使用正则表达式分割文本
        let mut result = Vec::new();
        for mat in self.base.compiled_pattern.find_iter(text) {
            let piece = match mat {
                Ok(m) => m.as_str(),
                Err(e) => {
                    return Err(crate::error::TokenizerError::EncodingError {
                        message: format!("正则表达式匹配失败: {}", e),
                    })
                }
            };

            if piece.is_empty() {
                continue;
            }

            // 首先尝试直接匹配整个片段 - O(1)查找
            let piece_string = piece.to_string();
            if let Some(&id) = self.vocab.get_by_value(&piece_string) {
                result.push(id);
                continue;
            }

            // 将文本转换为字符序列
            let mut ids: Vec<WordId> = Vec::new();
            for ch in piece.chars() {
                let ch_str = ch.to_string();
                // 使用反向映射进行O(1)查找
                if let Some(&id) = self.vocab.get_by_value(&ch_str) {
                    ids.push(id);
                } else {
                    // 如果找不到，使用字符的Unicode码点作为token ID
                    ids.push(ch as u32);
                }
            }

            // 应用合并规则 - 优化版本：贪心合并，避免重复扫描
            // 持续合并直到没有更多可以合并的对
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

                ids = new_ids;
            }

            result.extend(ids);
        }

        Ok(result)
    }

    /// 内部解码实现
    fn decode_internal(&self, tokens: Vec<u32>) -> Result<String, crate::error::TokenizerError> {
        let mut result = String::new();

        for token in tokens {
            if let Some(text) = self.vocab.get_by_id(&token) {
                // 直接使用词汇表中的文本
                result.push_str(text);
            } else {
                // 如果找不到对应的词汇，尝试作为Unicode字符处理
                if let Some(c) = char::from_u32(token) {
                    result.push(c);
                } else {
                    // 如果不是有效的Unicode字符，使用替换字符
                    result.push('�');
                }
            }
        }

        Ok(result)
    }
}

#[cfg(feature = "python")]
impl Default for Tokenizer {
    fn default() -> Self {
        Self::_new_internal().unwrap()
    }
}

#[cfg(feature = "python")]
impl TokenizerTrait for Tokenizer {
    type TokenId = u32;

    /// 编码文本为token ID序列
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self._encode_internal(text).map_err(|e| e.to_string())
    }

    /// 解码token ID序列为文本，参考template.rs中的实现
    fn decode(&self, tokens: &[u32]) -> Result<String, String> {
        self.decode_internal(tokens.to_vec())
            .map_err(|e| e.to_string())
    }

    /// 训练分词器，参考template.rs中的实现
    fn train(&mut self, texts: Vec<String>, vocab_size: u32) -> Result<(), String> {
        log::info!("开始BPE训练，目标词汇表大小: {}", vocab_size);

        // 确保词汇表大小不小于256
        let vocab_size = vocab_size.max(256);

        // 初始化合并规则
        self.merges.clear();

        // 将文本转换为词序列
        log::info!("处理 {} 个文本样本", texts.len());
        let words: Vec<Word<WordId>> = {
            let mut words = Vec::new();

            for text in &texts {
                // 使用正则表达式分割文本
                let parts = self.base.split_text(text)?;

                for part in parts {
                    if part.is_empty() {
                        continue;
                    }

                    // 将每个部分转换为字符序列
                    let mut ids = Vec::new();
                    for ch in part.chars() {
                        // 直接使用字符的Unicode码点作为token ID
                        let code_point = ch as u32;
                        // 确保字符在词汇表中
                        if !self.vocab.contains_id(&code_point) {
                            let ch_str = ch.to_string();
                            self.vocab.insert(code_point, ch_str);
                            if code_point >= self.next_token_id {
                                self.next_token_id = code_point + 1;
                            }
                        }
                        ids.push(code_point);
                    }

                    if !ids.is_empty() {
                        words.push(Word::new(ids));
                    }
                }
            }
            words
        };

        log::info!("已处理 {} 个词", words.len());

        // 使用增量训练核心
        self._train_core_incremental(words, vocab_size);
        log::info!("BPE训练完成，最终合并规则数: {}", self.merges.len());
        log::info!(
            "训练后词汇表大小: {}, next_token_id: {}",
            self.vocab.len(),
            self.next_token_id
        );

        Ok(())
    }

    fn vocab_size(&self) -> usize {
        log::debug!(
            "词汇表大小: {}, next_token_id: {}",
            self.vocab.len(),
            self.next_token_id
        );
        self.vocab.len()
    }

    fn save(&self, path: &str) -> Result<(), String> {
        // 使用基础分词器的保存方法
        self.base.save(path)?;

        // 保存BPE特定的数据
        use std::fs::OpenOptions;
        use std::io::Write;

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("打开文件失败: {}", e))?;

        // 保存词汇表
        writeln!(file, "vocab: {}", self.vocab.len())
            .map_err(|e| format!("写入词汇表大小失败: {}", e))?;

        for (&id, text) in self.vocab.iter() {
            writeln!(file, "vocab_entry: {} {}", id, text)
                .map_err(|e| format!("写入词汇表条目失败: {}", e))?;
        }

        // 保存合并规则
        writeln!(file, "merges: {}", self.merges.len())
            .map_err(|e| format!("写入合并规则数量失败: {}", e))?;

        for ((a, b), &new_id) in &self.merges {
            writeln!(file, "merge: {} {} {}", a, b, new_id)
                .map_err(|e| format!("写入合并规则失败: {}", e))?;
        }

        // 保存下一个可用的token ID
        writeln!(file, "next_token_id: {}", self.next_token_id)
            .map_err(|e| format!("写入下一个token ID失败: {}", e))?;

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<(), String> {
        // 使用基础分词器的加载方法
        self.base.load(path)?;

        // 加载BPE特定的数据
        use std::fs::File;
        use std::io::{self, BufRead};

        let file = File::open(path).map_err(|e| format!("打开文件失败: {}", e))?;
        let reader = io::BufReader::new(file);

        let lines = reader.lines();
        let mut in_vocab = false;
        let mut in_merges = false;

        // 清空当前数据
        self.vocab.clear();
        self.merges.clear();

        for line in lines {
            let line = line.map_err(|e| format!("读取行失败: {}", e))?;
            let line = line.trim();

            if line.starts_with("vocab: ") {
                in_vocab = true;
                continue;
            } else if line.starts_with("merges: ") {
                in_vocab = false;
                in_merges = true;
                continue;
            } else if line.starts_with("next_token_id: ") {
                let id_str = line[14..].trim();
                self.next_token_id = id_str
                    .parse::<WordId>()
                    .map_err(|e| format!("解析下一个token ID失败: {}", e))?;
            } else if line.starts_with("vocab_entry: ") {
                if in_vocab {
                    let parts: Vec<&str> = line[12..].splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        let id = parts[0]
                            .parse::<WordId>()
                            .map_err(|e| format!("解析词汇表ID失败: {}", e))?;
                        let text = parts[1].to_string();
                        self.vocab.insert(id, text);
                    }
                }
            } else if line.starts_with("merge: ") {
                if in_merges {
                    let parts: Vec<&str> = line[6..].split_whitespace().collect();
                    if parts.len() == 3 {
                        let a = parts[0]
                            .parse::<WordId>()
                            .map_err(|e| format!("解析合并规则a失败: {}", e))?;
                        let b = parts[1]
                            .parse::<WordId>()
                            .map_err(|e| format!("解析合并规则b失败: {}", e))?;
                        let new_id = parts[2]
                            .parse::<WordId>()
                            .map_err(|e| format!("解析合并规则new_id失败: {}", e))?;
                        self.merges.insert((a, b), new_id);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "python")]
impl MergeBasedTokenizer for Tokenizer {
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
