# WordPiece 算法文档

## 概述

WordPiece是一种基于概率的子词分词算法，最初由Google在日语和韩语语音识别系统中开发，后来被广泛应用于BERT等预训练语言模型中。与BPE不同，WordPiece不是基于频率合并字符对，而是选择能够最大化语言模型似然的子词合并。

## 算法原理

### 1. 基本思想

WordPiece的核心思想是选择能够最大化训练数据似然的子词。给定一个句子，WordPiece试图找到一种分词方式，使得：

```
P(sentence) = Π P(token)
```

其中，P(token)是每个token的概率。在训练过程中，WordPiece通过合并能够最大化整体似然的子词对来扩展词汇表。

### 2. 初始化

WordPiece算法从字符级别的词汇表开始：

1. 将所有训练数据中的字符作为初始词汇表
2. 添加特殊token（如[UNK]、[CLS]、[SEP]等）
3. 计算每个字符的概率

### 3. 迭代合并

算法通过迭代合并来扩展词汇表：

1. **生成候选合并**：生成所有可能的相邻子词对
2. **计算似然增益**：计算每个候选合并对整体似然的增益
3. **选择最佳合并**：选择能够最大化似然增益的子词对
4. **更新词汇表**：将选中的子词对添加到词汇表中
5. 重复上述步骤，直到达到目标词汇表大小

### 4. 分词策略

WordPiece使用贪心算法进行分词，优先选择概率最高的最长匹配：

```
给定句子S，从左到右：
1. 找到以当前位置开始的最长子词
2. 如果找到匹配，添加到结果中，移动到子词结束位置
3. 如果没有匹配，使用[UNK] token，移动一个字符位置
4. 重复直到处理完整个句子
```

## 核心实现

### 数据结构

1. **Tokenizer**：主要的分词器结构体，包含：
   - `vocab`：词汇表，映射子词到ID和概率
   - `unk_token`：未知词token
   - `max_token_length`：最大token长度

2. **Token**：token信息，包含：
   - `text`：token文本
   - `score`：token概率分数
   - `id`：token ID

3. **MergeCandidate**：合并候选，包含：
   - `pair`：子词对
   - `score`：合并后的似然增益

### 训练算法

```rust
fn train(&mut self, files: Vec<String>, vocab_size: usize, special_tokens: Vec<String>) {
    // 1. 初始化字符级词汇表
    self.initialize_char_vocab(&files);
    
    // 2. 读取训练数据
    let sentences = self.read_sentences(&files);
    
    // 3. 迭代合并直到达到目标词汇表大小
    while self.vocab.len() < vocab_size {
        // 3.1 生成所有可能的合并候选
        let candidates = self.generate_merge_candidates(&sentences);
        
        // 3.2 计算每个候选的似然增益
        let scored_candidates = self.score_merge_candidates(&candidates, &sentences);
        
        // 3.3 选择最佳合并
        if let Some(best_merge) = self.select_best_merge(&scored_candidates) {
            // 3.4 更新词汇表
            self.add_token_to_vocab(best_merge.token, best_merge.score);
        } else {
            break; // 没有更多可以合并的候选
        }
    }
    
    // 4. 添加特殊tokens
    self.add_special_tokens(special_tokens);
}
```

### 似然计算

```rust
fn calculate_likelihood(&self, sentences: &[Vec<String>]) -> f64 {
    let mut total_log_likelihood = 0.0;
    
    for sentence in sentences {
        let mut sentence_log_likelihood = 0.0;
        
        for token in sentence {
            if let Some(token_info) = self.vocab.get(token) {
                // 添加对数概率
                sentence_log_likelihood += token_info.score.ln();
            } else {
                // 未知词使用[UNK]的概率
                if let Some(unk_info) = self.vocab.get(&self.unk_token) {
                    sentence_log_likelihood += unk_info.score.ln();
                } else {
                    // 如果没有[UNK] token，使用一个很小的概率
                    sentence_log_likelihood += f64::EPSILON.ln();
                }
            }
        }
        
        total_log_likelihood += sentence_log_likelihood;
    }
    
    total_log_likelihood
}
```

### 分词算法

```rust
fn encode(&self, text: &str) -> Vec<usize> {
    // 1. 预处理文本
    let text = self.preprocess_text(text);
    
    // 2. 使用贪心算法进行分词
    let tokens = self.tokenize_greedy(&text);
    
    // 3. 转换为token ID序列
    tokens.iter().map(|token| {
        self.vocab.get(token)
            .map(|t| t.id)
            .unwrap_or_else(|| self.vocab[&self.unk_token].id)
    }).collect()
}

fn tokenize_greedy(&self, text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = text.chars().collect();
    
    while i < chars.len() {
        let mut found = false;
        
        // 从最长可能长度开始尝试匹配
        for length in (1..=self.max_token_length.min(chars.len() - i)).rev() {
            let token: String = chars[i..i+length].iter().collect();
            
            if self.vocab.contains_key(&token) {
                result.push(token);
                i += length;
                found = true;
                break;
            }
        }
        
        if !found {
            // 没有匹配的token，使用[UNK]
            result.push(self.unk_token.clone());
            i += 1;
        }
    }
    
    result
}
```

### 合并候选评分

```rust
fn score_merge_candidates(&self, candidates: &[MergeCandidate], sentences: &[Vec<String>]) -> Vec<MergeCandidate> {
    let mut scored_candidates = Vec::new();
    
    for candidate in candidates {
        // 1. 创建临时词汇表，包含新的合并token
        let mut temp_vocab = self.vocab.clone();
        let new_token = format!("{}{}", candidate.pair.0, candidate.pair.1);
        let new_score = self.calculate_new_token_score(&candidate.pair, sentences);
        temp_vocab.insert(new_token.clone(), TokenInfo {
            id: temp_vocab.len(),
            score: new_score,
        });
        
        // 2. 使用临时词汇表重新分词并计算似然
        let temp_sentences = self.retokenize_with_temp_vocab(sentences, &temp_vocab);
        let new_likelihood = self.calculate_likelihood_with_vocab(&temp_sentences, &temp_vocab);
        
        // 3. 计算似然增益
        let current_likelihood = self.calculate_likelihood(sentences);
        let likelihood_gain = new_likelihood - current_likelihood;
        
        scored_candidates.push(MergeCandidate {
            pair: candidate.pair.clone(),
            token: new_token,
            score: likelihood_gain,
        });
    }
    
    // 按似然增益排序
    scored_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    scored_candidates
}
```

## 算法特点

### 优点

1. **概率模型**：基于概率的选择，能够最大化语言模型似然
2. **灵活性**：可以处理各种语言的文本，包括低资源语言
3. **可控性**：通过设置词汇表大小可以精确控制模型复杂度
4. **有效性**：在预训练语言模型中表现出色

### 缺点

1. **计算复杂**：需要计算每个候选合并的似然增益，计算开销大
2. **贪心策略**：分词时使用贪心策略，可能不是全局最优
3. **训练时间长**：迭代过程需要较长时间

## 应用场景

1. **预训练语言模型**：BERT、RoBERTa等模型使用WordPiece作为分词器
2. **多语言处理**：能够处理多种语言的混合文本
3. **领域适应**：可以适应特定领域的术语和表达
4. **低资源语言**：对于低资源语言，可以更好地利用有限的训练数据

## 性能优化

1. **高效数据结构**：使用前缀树(Trie)优化子词匹配
2. **并行计算**：并行处理不同句子的分词和似然计算
3. **缓存机制**：缓存频繁使用的子词和概率
4. **剪枝策略**：使用启发式方法减少候选数量

## 与其他算法比较

| 算法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| BPE | 基于频率的字符对合并 | 简单高效，易于控制词汇表大小 | 贪心策略可能不是最优 |
| WordPiece | 基于似然最大化的子词选择 | 考虑了整个序列的概率 | 训练更复杂 |
| Unigram | 基于概率的语言模型 | 灵活性高，支持多种分词策略 | 需要更多计算资源 |
| BBPE | 字节级别的BPE | 支持任意Unicode字符 | 词汇表可能更大 |

## 实际应用案例

### BERT中的WordPiece

BERT使用了WordPiece算法进行分词：

1. **预处理**：文本规范化、添加特殊标记
2. **词汇表**：30,000个token的词汇表
3. **特殊标记**：[CLS]、[SEP]、[PAD]、[MASK]、[UNK]
4. **分词策略**：最长匹配优先，未知词使用[UNK]

### 处理示例

```
文本: "Hello, world!"
分词过程:
1. 初始: ["H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"]
2. 合并 "l" + "o" → "lo" (似然增益最高)
3. 合并 "e" + "lo" → "elo" (似然增益最高)
4. 合并 "H" + "elo" → "Hello" (似然增益最高)
5. 合并 "w" + "o" → "wo" (似然增益最高)
6. 合并 "wo" + "r" → "wor" (似然增益最高)
7. 合并 "wor" + "l" → "worl" (似然增益最高)
8. 合并 "worl" + "d" → "world" (似然增益最高)

最终分词: ["Hello", ",", " ", "world", "!"]
```