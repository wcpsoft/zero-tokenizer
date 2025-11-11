/// 通用的词表示（ID序列）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word<Id> {
    /// ID序列
    pub ids: Vec<Id>,
}

impl<Id> Word<Id> {
    /// 创建新的词
    pub fn new(ids: Vec<Id>) -> Self {
        Self { ids }
    }
    
    /// 获取ID序列
    pub fn ids(&self) -> &[Id] {
        &self.ids
    }
    
    /// 获取ID序列的可变引用
    pub fn ids_mut(&mut self) -> &mut Vec<Id> {
        &mut self.ids
    }
}

impl<Id: Clone> Word<Id> {
    /// 获取所有相邻的ID对
    pub fn pairs(&self) -> impl Iterator<Item = (Id, Id)> + '_ {
        self.ids.windows(2).map(|w| (w[0].clone(), w[1].clone()))
    }
    
    /// 合并指定的词对，返回受影响的词对及其变化
    /// 这是一个通用实现，适用于BPE和BBPE
    pub fn merge_pair<F>(&mut self, pair: (Id, Id), new_id: Id, id_eq: F) -> Vec<((Id, Id), i32)>
    where
        F: Fn(&Id, &Id) -> bool,
        Id: PartialEq,
    {
        let mut affected_pairs = Vec::new();
        let mut i = 0;
        
        while i < self.ids.len() - 1 {
            if id_eq(&self.ids[i], &pair.0) && id_eq(&self.ids[i + 1], &pair.1) {
                // 记录受影响的词对
                if i > 0 {
                    affected_pairs.push(((self.ids[i - 1].clone(), self.ids[i].clone()), -1));
                }
                if i + 2 < self.ids.len() {
                    affected_pairs.push(((self.ids[i + 1].clone(), self.ids[i + 2].clone()), -1));
                }
                
                // 执行合并
                self.ids[i] = new_id.clone();
                self.ids.remove(i + 1);
                
                // 记录新创建的词对
                if i > 0 {
                    affected_pairs.push(((self.ids[i - 1].clone(), self.ids[i].clone()), 1));
                }
                if i < self.ids.len() - 1 {
                    affected_pairs.push(((self.ids[i].clone(), self.ids[i + 1].clone()), 1));
                }
            } else {
                i += 1;
            }
        }
        
        affected_pairs
    }
}