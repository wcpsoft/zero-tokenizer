use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::Hash;

/// 表示一个合并任务
#[derive(Debug, Clone)]
pub struct MergeJob<Id: Ord> {
    /// 要合并的词对
    pub pair: (Id, Id),
    /// 词对出现的次数
    pub count: u64,
    /// 需要处理此配对的词索引集合
    pub pos: HashSet<usize>,
}

impl<Id: PartialEq + Ord> PartialEq for MergeJob<Id> {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl<Id: Eq + Ord> Eq for MergeJob<Id> {}

impl<Id: PartialOrd + Ord> PartialOrd for MergeJob<Id> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Id: Ord> Ord for MergeJob<Id> {
    fn cmp(&self, other: &Self) -> Ordering {
        // 按计数最大堆；计数相同时按配对升序（确定性）
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // 计数相同时按配对升序
            other.pair.cmp(&self.pair)
        }
    }
}

impl<Id: Clone + Hash + Eq + Ord> MergeJob<Id> {
    /// 创建新的合并任务
    pub fn new(pair: (Id, Id), count: u64) -> Self {
        Self {
            pair,
            count,
            pos: HashSet::new(),
        }
    }

    /// 添加词索引
    pub fn add_position(&mut self, pos: usize) {
        self.pos.insert(pos);
    }

    /// 批量添加词索引
    pub fn add_positions(&mut self, positions: &[usize]) {
        for &pos in positions {
            self.pos.insert(pos);
        }
    }
}
