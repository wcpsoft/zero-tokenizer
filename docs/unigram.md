# Unigram 语言模型算法文档

## 概述

Unigram语言模型是一种基于概率的子词分词方法，也称为子词正则分词。与BPE和WordPiece不同，Unigram不是通过迭代合并来构建词汇表，而是从一个大的初始词汇表开始，逐步移除不重要的子词，直到达到目标词汇表大小。这种方法最初由SentencePiece模型引入，被广泛应用于各种NLP任务中。

## 算法原理

### 1. 基本思想

Unigram模型假设每个子词的出现是独立的，因此一个句子（子词序列）的概率可以表示为各个子词概率的乘积：

```
P(x1, x2, ..., xn) = Π P(xi)
```

其中，P(xi)是子词xi的概率，通过在训练语料中的频率计算得出。

### 2. 初始化

Unigram算法从一个大的初始词汇表开始，通常包含：

1. 所有训练语料中的字符
2. 常见的子词组合
3. 预定义的特殊token

初始词汇表大小通常远大于目标词汇表。

### 3. 迭代剪枝

算法通过迭代移除对模型似然影响最小的子词来缩小词汇表：

1. **E步（期望）**：使用当前词汇表对所有训练数据进行分词
2. **M步（最大化）**：计算每个子词的概率，并评估移除每个子词对整体似然的影响
3. **剪枝**：移除对整体似然影响最小的子词
4. 重复上述步骤，直到达到目标词汇表大小

### 4. 分词策略

Unigram使用Viterbi算法找到概率最大的子词序列：

```
给定句子S，找到子词序列T = (t1, t2, ..., tn)，使得：
P(T|S) = Π P(ti) 最大
```

## 核心实现

### 数据结构

1. **Tokenizer**：主要的分词器结构体，包含：
   - `vocab`：词汇表，映射子词到ID和概率
   - `unk_token`：未知词token
   - `score_threshold`：分数阈值，用于过滤低概率子词

2. **Subword**：子词信息，包含：
   - `text`：子词文本
   - `score`：子词概率分数
   - `id`：子词ID

3. **Lattice**：格点结构，用于Viterbi算法：
   - `nodes`：格点节点，表示可能的子词分割
   - `edges`：格点边，表示子词之间的连接

### 训练算法

```rust
fn train(&mut self, files: Vec<String>, vocab_size: usize, special_tokens: Vec<String>) {
    // 1. 初始化大词汇表
    self.initialize_large_vocab(&files);
    
    // 2. 读取训练数据
    let sentences = self.read_sentences(&files);
    
    // 3. 迭代剪枝直到达到目标词汇表大小
    while self.vocab.len() > vocab_size {
        // 3.1 E步：使用当前词汇表对所有训练数据进行分词
        let segmented_sentences = self.segment_all_sentences(&sentences);
        
        // 3.2 M步：计算每个子词的概率
        self.update_subword_scores(&segmented_sentences);
        
        // 3.3 计算移除每个子词对整体似然的影响
        let loss_changes = self.calculate_removal_impact(&segmented_sentences);
        
        // 3.4 剪枝：移除对整体似然影响最小的子词
        let num_to_remove = (self.vocab.len() - vocab_size).min(self.vocab.len() / 10);
        self.prune_vocab(&loss_changes, num_to_remove);
    }
    
    // 4. 添加特殊tokens
    self.add_special_tokens(special_tokens);
}
```

### Viterbi分词算法

```rust
fn encode(&self, text: &str) -> Vec<usize> {
    // 1. 预处理文本
    let text = self.preprocess_text(text);
    
    // 2. 使用Viterbi算法找到最优分词
    let lattice = self.build_lattice(&text);
    let best_path = self.viterbi_decode(&lattice);
    
    // 3. 转换为token ID序列
    best_path.iter().map(|&subword| self.vocab[subword].id).collect()
}

fn viterbi_decode(&self, lattice: &Lattice) -> Vec<String> {
    let n = lattice.len();
    let mut best_scores = vec![f64::NEG_INFINITY; n + 1];
    let mut backpointers = vec![None; n + 1];
    
    best_scores[0] = 0.0; // 初始状态分数为0
    
    // 动态规划填充格点
    for i in 0..n {
        if best_scores[i] == f64::NEG_INFINITY {
            continue; // 不可达状态
        }
        
        // 遍历从位置i开始的所有可能子词
        for (j, subword) in lattice.get_subwords_starting_at(i) {
            let score = best_scores[i] + subword.score.log();
            if score > best_scores[j] {
                best_scores[j] = score;
                backpointers[j] = Some((i, subword.text.clone()));
            }
        }
    }
    
    // 回溯找到最优路径
    let mut result = Vec::new();
    let mut pos = n;
    while pos > 0 {
        if let Some((prev_pos, subword)) = &backpointers[pos] {
            result.push(subword.clone());
            pos = *prev_pos;
        } else {
            // 处理未知词
            result.push(self.unk_token.clone());
            pos -= 1;
        }
    }
    
    result.reverse(); // 反转得到正确顺序
    result
}
```

### 子词分数更新

```rust
fn update_subword_scores(&mut self, segmented_sentences: &[Vec<String>]) {
    // 1. 统计每个子词的频率
    let mut frequencies = HashMap::new();
    for sentence in segmented_sentences {
        for subword in sentence {
            *frequencies.entry(subword.clone()).or_insert(0) += 1;
        }
    }
    
    // 2. 计算总频率
    let total_frequency: u32 = frequencies.values().sum();
    
    // 3. 更新每个子词的概率分数
    for (subword, freq) in frequencies {
        if let Some(token) = self.vocab.get_mut(&subword) {
            token.score = freq as f64 / total_frequency as f64;
        }
    }
}
```

## 算法特点

### 优点

1. **概率模型**：基于概率的模型，可以量化每个子词的重要性
2. **灵活性高**：可以处理多种分词策略，如最长匹配、最优匹配等
3. **语言无关性**：不依赖于特定语言的语言学知识
4. **可解释性**：每个子词都有明确的概率分数

### 缺点

1. **计算复杂**：需要多次迭代和Viterbi算法，计算开销较大
2. **内存需求高**：需要存储大的初始词汇表和中间结果
3. **训练时间长**：迭代剪枝过程需要较长时间

## 应用场景

1. **多语言处理**：特别适合处理多语言混合的文本
2. **低资源语言**：对于低资源语言，可以更好地利用有限的训练数据
3. **领域适应**：可以快速适应特定领域的术语和表达
4. **语音识别**：在语音识别中处理发音变体和口语化表达

## 性能优化

1. **高效数据结构**：使用前缀树(Trie)优化子词匹配
2. **并行计算**：并行处理不同句子的分词
3. **增量更新**：支持增量式更新子词概率
4. **剪枝策略**：使用更智能的剪枝策略减少迭代次数

## 与其他算法比较

| 算法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| BPE | 基于频率的字符对合并 | 简单高效，易于控制词汇表大小 | 贪心策略可能不是最优 |
| WordPiece | 基于似然最大化的子词选择 | 考虑了整个序列的概率 | 训练更复杂 |
| Unigram | 基于概率的语言模型 | 灵活性高，支持多种分词策略 | 需要更多计算资源 |
| BBPE | 字节级别的BPE | 支持任意Unicode字符 | 词汇表可能更大 |

## 实际应用案例

### SentencePiece中的Unigram

SentencePiece是一个开源的子词分词工具包，实现了Unigram算法：

1. **预处理**：将文本规范化并添加特殊标记
2. **初始词汇表**：从训练数据中提取所有可能的子词
3. **训练过程**：使用EM算法迭代优化子词概率
4. **分词策略**：使用Viterbi算法找到最优分词

### 处理示例

```
文本: "Hello, world!"
初始分词: ["H", "e", "l", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"]
优化分词: ["Hello", ",", " ", "world", "!"]
子词概率: {"Hello": 0.01, "world": 0.008, ",": 0.05, "!": 0.03}
```
