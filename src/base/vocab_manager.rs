use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// 通用词汇表管理器，封装双向映射的同步管理
///
/// 解决问题：
/// - 自动维护正向和反向映射的一致性
/// - 防止手动同步导致的bug
/// - 提供类型安全的操作
///
/// # 类型参数
/// - `K`: Token ID类型（如u32, usize）
/// - `V`: Token值类型（如String, Vec<u8>）
///
/// # 示例
/// ```
/// use zero_tokenizer::base::vocab_manager::VocabManager;
///
/// let mut vocab = VocabManager::<u32, String>::new();
/// vocab.insert(0, "hello".to_string());
///
/// assert_eq!(vocab.get_by_id(&0), Some(&"hello".to_string()));
/// assert_eq!(vocab.get_by_value("hello"), Some(&0));
/// ```
#[derive(Clone, Debug)]
pub struct VocabManager<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Eq + Hash + Clone + Debug,
{
    /// 正向映射: ID -> Value
    id_to_value: HashMap<K, V>,
    /// 反向映射: Value -> ID
    value_to_id: HashMap<V, K>,
}

impl<K, V> VocabManager<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Eq + Hash + Clone + Debug,
{
    /// 创建新的空词汇表管理器
    pub fn new() -> Self {
        Self {
            id_to_value: HashMap::new(),
            value_to_id: HashMap::new(),
        }
    }

    /// 使用指定容量创建词汇表管理器
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            id_to_value: HashMap::with_capacity(capacity),
            value_to_id: HashMap::with_capacity(capacity),
        }
    }

    /// 插入一个token，自动维护双向映射
    ///
    /// # 返回值
    /// - `Some(old_value)`: 如果ID已存在，返回旧值
    /// - `None`: 新插入
    ///
    /// # 注意
    /// 如果value已存在但ID不同，会覆盖旧的映射
    pub fn insert(&mut self, id: K, value: V) -> Option<V> {
        // 首先，如果这个ID已存在，移除其旧值的反向映射
        if let Some(old_value) = self.id_to_value.get(&id) {
            if old_value != &value {
                self.value_to_id.remove(old_value);
            }
        }

        // 然后，如果新值已存在于其他ID，移除那个旧的ID映射
        if let Some(old_id) = self.value_to_id.get(&value) {
            if old_id != &id {
                self.id_to_value.remove(old_id);
            }
        }

        // 最后，插入新映射
        self.value_to_id.insert(value.clone(), id.clone());
        self.id_to_value.insert(id, value)
    }

    /// 根据ID获取值
    #[inline]
    pub fn get_by_id(&self, id: &K) -> Option<&V> {
        self.id_to_value.get(id)
    }

    /// 根据值获取ID
    #[inline]
    pub fn get_by_value(&self, value: &V) -> Option<&K> {
        self.value_to_id.get(value)
    }

    /// 检查ID是否存在
    #[inline]
    pub fn contains_id(&self, id: &K) -> bool {
        self.id_to_value.contains_key(id)
    }

    /// 检查值是否存在
    #[inline]
    pub fn contains_value(&self, value: &V) -> bool {
        self.value_to_id.contains_key(value)
    }

    /// 根据ID移除token
    ///
    /// # 返回值
    /// 返回被移除的值（如果存在）
    pub fn remove_by_id(&mut self, id: &K) -> Option<V> {
        if let Some(value) = self.id_to_value.remove(id) {
            self.value_to_id.remove(&value);
            Some(value)
        } else {
            None
        }
    }

    /// 根据值移除token
    ///
    /// # 返回值
    /// 返回被移除的ID（如果存在）
    pub fn remove_by_value(&mut self, value: &V) -> Option<K> {
        if let Some(id) = self.value_to_id.remove(value) {
            self.id_to_value.remove(&id);
            Some(id)
        } else {
            None
        }
    }

    /// 清空所有映射
    pub fn clear(&mut self) {
        self.id_to_value.clear();
        self.value_to_id.clear();
    }

    /// 获取词汇表大小
    #[inline]
    pub fn len(&self) -> usize {
        self.id_to_value.len()
    }

    /// 检查词汇表是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id_to_value.is_empty()
    }

    /// 获取所有ID的迭代器
    pub fn ids(&self) -> impl Iterator<Item = &K> {
        self.id_to_value.keys()
    }

    /// 获取所有值的迭代器
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.id_to_value.values()
    }

    /// 获取所有(ID, Value)对的迭代器
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.id_to_value.iter()
    }

    /// 获取正向映射的引用（ID -> Value）
    ///
    /// # 用途
    /// 用于需要直接访问HashMap的场景（如序列化）
    pub fn id_map(&self) -> &HashMap<K, V> {
        &self.id_to_value
    }

    /// 获取反向映射的引用（Value -> ID）
    ///
    /// # 用途
    /// 用于需要直接访问HashMap的场景（如序列化）
    pub fn value_map(&self) -> &HashMap<V, K> {
        &self.value_to_id
    }

    /// 验证双向映射的一致性
    ///
    /// # 返回值
    /// - `Ok(())`: 映射一致
    /// - `Err(String)`: 发现不一致，返回错误信息
    ///
    /// # 用途
    /// 用于调试和测试，确保数据完整性
    pub fn validate(&self) -> Result<(), String> {
        // 检查大小一致性
        if self.id_to_value.len() != self.value_to_id.len() {
            return Err(format!(
                "Size mismatch: id_to_value={}, value_to_id={}",
                self.id_to_value.len(),
                self.value_to_id.len()
            ));
        }

        // 检查每个正向映射都有对应的反向映射
        for (id, value) in &self.id_to_value {
            match self.value_to_id.get(value) {
                Some(reverse_id) if reverse_id == id => {}
                Some(reverse_id) => {
                    return Err(format!(
                        "Reverse mapping mismatch: id={:?} maps to value={:?}, \
                         but value maps back to different id={:?}",
                        id, value, reverse_id
                    ));
                }
                None => {
                    return Err(format!(
                        "Missing reverse mapping: id={:?} -> value={:?}",
                        id, value
                    ));
                }
            }
        }

        Ok(())
    }

    /// 从现有的HashMap构建VocabManager
    ///
    /// # 参数
    /// - `id_to_value`: 正向映射
    ///
    /// # 注意
    /// 会自动构建反向映射
    pub fn from_id_map(id_to_value: HashMap<K, V>) -> Self {
        let value_to_id = id_to_value
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();

        Self {
            id_to_value,
            value_to_id,
        }
    }
}

impl<K, V> Default for VocabManager<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Eq + Hash + Clone + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// 实现迭代器trait
impl<K, V> IntoIterator for VocabManager<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Eq + Hash + Clone + Debug,
{
    type Item = (K, V);
    type IntoIter = std::collections::hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.id_to_value.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut vocab = VocabManager::<u32, String>::new();

        // 插入
        assert_eq!(vocab.insert(0, "hello".to_string()), None);
        assert_eq!(vocab.insert(1, "world".to_string()), None);

        // 查询
        assert_eq!(vocab.get_by_id(&0), Some(&"hello".to_string()));
        assert_eq!(vocab.get_by_value(&"hello".to_string()), Some(&0));

        // 大小
        assert_eq!(vocab.len(), 2);
        assert!(!vocab.is_empty());
    }

    #[test]
    fn test_overwrite() {
        let mut vocab = VocabManager::<u32, String>::new();

        vocab.insert(0, "hello".to_string());
        vocab.insert(0, "world".to_string()); // 覆盖

        assert_eq!(vocab.get_by_id(&0), Some(&"world".to_string()));
        assert_eq!(vocab.get_by_value(&"world".to_string()), Some(&0));
        assert_eq!(vocab.get_by_value(&"hello".to_string()), None); // 旧值应被移除
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut vocab = VocabManager::<u32, String>::new();

        vocab.insert(0, "hello".to_string());
        vocab.insert(1, "world".to_string());

        // 根据ID移除
        assert_eq!(vocab.remove_by_id(&0), Some("hello".to_string()));
        assert_eq!(vocab.get_by_id(&0), None);
        assert_eq!(vocab.get_by_value(&"hello".to_string()), None);

        // 根据值移除
        assert_eq!(vocab.remove_by_value(&"world".to_string()), Some(1));
        assert_eq!(vocab.len(), 0);
    }

    #[test]
    fn test_validation() {
        let mut vocab = VocabManager::<u32, String>::new();

        vocab.insert(0, "hello".to_string());
        vocab.insert(1, "world".to_string());

        // 正常情况应该验证通过
        assert!(vocab.validate().is_ok());
    }

    #[test]
    fn test_from_id_map() {
        let mut map = HashMap::new();
        map.insert(0u32, "hello".to_string());
        map.insert(1u32, "world".to_string());

        let vocab = VocabManager::from_id_map(map);

        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.get_by_id(&0), Some(&"hello".to_string()));
        assert_eq!(vocab.get_by_value(&"world".to_string()), Some(&1));
        assert!(vocab.validate().is_ok());
    }

    #[test]
    fn test_with_vec_u8() {
        let mut vocab = VocabManager::<u32, Vec<u8>>::new();

        vocab.insert(0, vec![72, 105]); // "Hi"
        vocab.insert(1, vec![66, 121]); // "By"

        assert_eq!(vocab.get_by_id(&0), Some(&vec![72, 105]));
        assert_eq!(vocab.get_by_value(&vec![66, 121]), Some(&1));
        assert!(vocab.validate().is_ok());
    }
}
