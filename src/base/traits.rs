use std::collections::HashMap;

/// 分词器基础接口，定义所有分词器必须实现的方法
pub trait Tokenizer {
    /// 标记ID类型
    type TokenId;

    /// 编码文本为标记ID序列
    ///
    /// # Errors
    ///
    /// 当正则表达式分割失败或文本包含无效字符时返回错误
    fn encode(&self, text: &str) -> Result<Vec<Self::TokenId>, String>;

    /// 解码标记ID序列为文本
    ///
    /// # Errors
    ///
    /// 当标记ID不在词汇表中时返回错误
    fn decode(&self, tokens: &[Self::TokenId]) -> Result<String, String>;

    /// 训练分词器
    ///
    /// # Errors
    ///
    /// 当训练文本为空、词汇表大小无效或训练过程中出现错误时返回错误
    fn train(&mut self, texts: Vec<String>, vocab_size: u32) -> Result<(), String>;

    /// 获取词汇表大小
    fn vocab_size(&self) -> usize;

    /// 保存分词器到文件
    ///
    /// # Errors
    ///
    /// 当文件无法创建、写入失败或路径无效时返回错误
    fn save(&self, path: &str) -> Result<(), String>;

    /// 从文件加载分词器
    ///
    /// # Errors
    ///
    /// 当文件不存在、格式无效或解析失败时返回错误
    fn load(&mut self, path: &str) -> Result<(), String>;
}

/// 基于合并的分词器接口（BPE和BBPE）
pub trait MergeBasedTokenizer: Tokenizer {
    /// 应用合并规则到标记序列
    ///
    /// # Errors
    ///
    /// 当标记序列无效或合并规则应用失败时返回错误
    fn apply_merges(&mut self, tokens: &mut Vec<Self::TokenId>) -> Result<(), String>;

    /// 获取合并规则
    fn get_merges(&self) -> &HashMap<(Self::TokenId, Self::TokenId), Self::TokenId>;

    /// 设置合并规则
    fn set_merges(&mut self, merges: HashMap<(Self::TokenId, Self::TokenId), Self::TokenId>);
}

/// 基于子词的分词器接口（WordPiece和Unigram）
pub trait SubwordTokenizer: Tokenizer {
    /// 获取标记分数（用于Unigram）
    fn get_scores(&self) -> Option<&Vec<f64>>;

    /// 设置标记分数（用于Unigram）
    fn set_scores(&mut self, scores: Vec<f64>);
}
