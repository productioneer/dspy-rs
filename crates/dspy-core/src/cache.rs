//! DSPy Cache — Two-tier caching system for LM responses.
//!
//! Level 1: In-memory LRU cache (fast, bounded by entry count)
//! Level 2: On-disk cache via JSON files (persistent, bounded by size)
//!
//! Matches Python DSPy's dspy.cache interface.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// Simple LRU cache backed by a Vec for ordering + HashMap for lookup.
struct LRUCache {
    max_size: usize,
    order: Vec<String>,
    map: HashMap<String, serde_json::Value>,
}

impl LRUCache {
    fn new(max_size: usize) -> Self {
        Self {
            max_size,
            order: Vec::new(),
            map: HashMap::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<&serde_json::Value> {
        if self.map.contains_key(key) {
            // Move to end (most recently used)
            self.order.retain(|k| k != key);
            self.order.push(key.to_string());
            self.map.get(key)
        } else {
            None
        }
    }

    fn set(&mut self, key: String, value: serde_json::Value) {
        if self.map.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else if self.order.len() >= self.max_size {
            // Evict least recently used
            if let Some(lru_key) = self.order.first().cloned() {
                self.order.remove(0);
                self.map.remove(&lru_key);
            }
        }
        self.order.push(key.clone());
        self.map.insert(key, value);
    }

    fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    fn clear(&mut self) {
        self.order.clear();
        self.map.clear();
    }
}

/// Metadata for a cache file used during eviction.
struct CacheFileInfo {
    path: PathBuf,
    size: usize,
    modified: std::time::SystemTime,
}

/// Disk-based cache using a directory of sharded JSON files.
struct DiskCache {
    dir: PathBuf,
    size_limit_bytes: usize,
}

impl DiskCache {
    fn new(dir: &str, size_limit_bytes: usize) -> Self {
        let path = PathBuf::from(dir);
        if !path.exists() {
            let _ = std::fs::create_dir_all(&path);
        }
        Self {
            dir: path,
            size_limit_bytes,
        }
    }

    fn key_path(&self, key: &str) -> PathBuf {
        // Use first 2 chars as shard directory
        let shard = &key[..2.min(key.len())];
        self.dir.join(shard).join(format!("{}.json", key))
    }

    fn contains(&self, key: &str) -> bool {
        self.key_path(key).exists()
    }

    fn get(&self, key: &str) -> Option<serde_json::Value> {
        let path = self.key_path(key);
        if !path.exists() {
            return None;
        }
        let data = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    }

    fn set(&self, key: &str, value: &serde_json::Value) {
        let path = self.key_path(key);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(data) = serde_json::to_string(value) {
            // Skip if single item exceeds limit
            if data.len() > self.size_limit_bytes {
                return;
            }
            // Evict oldest files if total size would exceed limit
            self.evict_if_needed(data.len());
            let _ = std::fs::write(&path, data);
        }
    }

    /// Evict oldest cache files (by mtime) until there's room for a new entry.
    /// Matches Python DSPy's diskcache behavior of enforcing total size limits.
    fn evict_if_needed(&self, new_entry_bytes: usize) {
        let mut files = match self.list_cache_files() {
            Ok(f) => f,
            Err(_) => return,
        };

        let mut total_size: usize = files.iter().map(|f| f.size).sum();

        if total_size + new_entry_bytes <= self.size_limit_bytes {
            return;
        }

        // Sort by mtime ascending (oldest first)
        files.sort_by(|a, b| a.modified.cmp(&b.modified));

        for file in &files {
            if total_size + new_entry_bytes <= self.size_limit_bytes {
                break;
            }
            if std::fs::remove_file(&file.path).is_ok() {
                total_size = total_size.saturating_sub(file.size);
            }
        }
    }

    /// List all .json files in the cache directory with size and mtime.
    fn list_cache_files(&self) -> std::io::Result<Vec<CacheFileInfo>> {
        let mut results = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if !meta.is_dir() {
                continue;
            }
            // Read shard directory
            for file_entry in std::fs::read_dir(entry.path())? {
                let file_entry = file_entry?;
                let file_name = file_entry.file_name();
                let name = file_name.to_string_lossy();
                if !name.ends_with(".json") {
                    continue;
                }
                if let Ok(file_meta) = file_entry.metadata() {
                    let modified = file_meta
                        .modified()
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    results.push(CacheFileInfo {
                        path: file_entry.path(),
                        size: file_meta.len() as usize,
                        modified,
                    });
                }
            }
        }
        Ok(results)
    }
}

/// Cache configuration options.
pub struct CacheConfig {
    /// Enable in-memory LRU cache. Default: true
    pub enable_memory_cache: bool,
    /// Enable on-disk persistent cache. Default: false
    pub enable_disk_cache: bool,
    /// Directory for disk cache storage. Default: ".dspy_cache"
    pub disk_cache_dir: String,
    /// Max size of disk cache in bytes. Default: 10MB
    pub disk_size_limit_bytes: usize,
    /// Max entries in memory cache. Default: 1_000_000
    pub memory_max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_memory_cache: true,
            enable_disk_cache: false,
            disk_cache_dir: ".dspy_cache".to_string(),
            disk_size_limit_bytes: 10 * 1024 * 1024,
            memory_max_entries: 1_000_000,
        }
    }
}

/// Default ignored arguments for cache key generation.
const DEFAULT_IGNORED_ARGS: &[&str] = &["api_key", "api_base", "base_url"];

/// DSPy Cache — two-tier caching (memory LRU + disk).
pub struct Cache {
    enable_memory_cache: bool,
    enable_disk_cache: bool,
    memory_cache: Mutex<LRUCache>,
    disk_cache: Option<DiskCache>,
}

impl Cache {
    /// Create a new Cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        assert!(
            config.memory_max_entries > 0,
            "memory_max_entries must be positive"
        );

        let disk_cache = if config.enable_disk_cache {
            Some(DiskCache::new(
                &config.disk_cache_dir,
                config.disk_size_limit_bytes,
            ))
        } else {
            None
        };

        Self {
            enable_memory_cache: config.enable_memory_cache,
            enable_disk_cache: config.enable_disk_cache,
            memory_cache: Mutex::new(LRUCache::new(config.memory_max_entries)),
            disk_cache,
        }
    }

    /// Create a cache with default settings (memory only).
    pub fn default_memory_only() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Generate a cache key from a request by hashing its JSON representation.
    pub fn cache_key(&self, request: &serde_json::Value, ignored_args: Option<&[&str]>) -> String {
        let ignored: &[&str] = ignored_args.unwrap_or(DEFAULT_IGNORED_ARGS);

        let filtered = self.sort_value_keys(request, ignored);

        let json = serde_json::to_string(&filtered).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Recursively sort object keys for deterministic hashing.
    fn sort_value_keys(&self, value: &serde_json::Value, ignored: &[&str]) -> serde_json::Value {
        match value {
            serde_json::Value::Object(obj) => {
                let mut sorted: Vec<(&String, &serde_json::Value)> = obj
                    .iter()
                    .filter(|(k, _)| !ignored.contains(&k.as_str()))
                    .collect();
                sorted.sort_by_key(|(k, _)| k.as_str());
                let map: serde_json::Map<String, serde_json::Value> = sorted
                    .into_iter()
                    .map(|(k, v)| (k.clone(), self.sort_value_keys(v, &[])))
                    .collect();
                serde_json::Value::Object(map)
            }
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(|v| self.sort_value_keys(v, &[])).collect())
            }
            _ => value.clone(),
        }
    }

    /// Check if a key exists in either cache tier.
    pub fn contains(&self, key: &str) -> bool {
        if self.enable_memory_cache {
            if let Ok(cache) = self.memory_cache.lock() {
                if cache.contains(key) {
                    return true;
                }
            }
        }
        if self.enable_disk_cache {
            if let Some(ref dc) = self.disk_cache {
                if dc.contains(key) {
                    return true;
                }
            }
        }
        false
    }

    /// Get a cached response by request object.
    pub fn get(
        &self,
        request: &serde_json::Value,
        ignored_args: Option<&[&str]>,
    ) -> Option<serde_json::Value> {
        if !self.enable_memory_cache && !self.enable_disk_cache {
            return None;
        }

        let key = self.cache_key(request, ignored_args);

        // Check memory cache first
        if self.enable_memory_cache {
            if let Ok(mut cache) = self.memory_cache.lock() {
                if let Some(value) = cache.get(&key) {
                    let mut response = value.clone();
                    // Clear usage data on cache hit
                    if let Some(obj) = response.as_object_mut() {
                        obj.remove("usage");
                        obj.insert("cache_hit".to_string(), serde_json::Value::Bool(true));
                    }
                    return Some(response);
                }
            }
        }

        // Check disk cache
        if self.enable_disk_cache {
            if let Some(ref dc) = self.disk_cache {
                if let Some(value) = dc.get(&key) {
                    // Promote to memory cache
                    if self.enable_memory_cache {
                        if let Ok(mut cache) = self.memory_cache.lock() {
                            cache.set(key, value.clone());
                        }
                    }
                    let mut response = value;
                    if let Some(obj) = response.as_object_mut() {
                        obj.remove("usage");
                        obj.insert("cache_hit".to_string(), serde_json::Value::Bool(true));
                    }
                    return Some(response);
                }
            }
        }

        None
    }

    /// Store a value in the cache.
    pub fn put(
        &self,
        request: &serde_json::Value,
        value: &serde_json::Value,
        ignored_args: Option<&[&str]>,
        enable_memory: bool,
    ) {
        let use_memory = self.enable_memory_cache && enable_memory;
        if !use_memory && !self.enable_disk_cache {
            return;
        }

        let key = self.cache_key(request, ignored_args);

        if use_memory {
            if let Ok(mut cache) = self.memory_cache.lock() {
                cache.set(key.clone(), value.clone());
            }
        }

        if self.enable_disk_cache {
            if let Some(ref dc) = self.disk_cache {
                dc.set(&key, value);
            }
        }
    }

    /// Clear the in-memory cache.
    pub fn reset_memory_cache(&self) {
        if let Ok(mut cache) = self.memory_cache.lock() {
            cache.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_memory_hit() {
        let cache = Cache::default_memory_only();
        let request = serde_json::json!({"model": "test", "prompt": "hello"});
        let response = serde_json::json!({"text": "world", "usage": {"tokens": 5}});

        assert!(cache.get(&request, None).is_none());

        cache.put(&request, &response, None, true);

        let result = cache.get(&request, None).unwrap();
        assert_eq!(result["text"], "world");
        assert_eq!(result["cache_hit"], true);
        // usage should be cleared on cache hit
        assert!(result.get("usage").is_none());
    }

    #[test]
    fn test_cache_key_deterministic() {
        let cache = Cache::default_memory_only();
        let req1 = serde_json::json!({"a": 1, "b": 2});
        let req2 = serde_json::json!({"b": 2, "a": 1});
        // Keys should be the same regardless of object key order
        assert_eq!(cache.cache_key(&req1, None), cache.cache_key(&req2, None));
    }

    #[test]
    fn test_cache_key_ignores_auth() {
        let cache = Cache::default_memory_only();
        let req1 = serde_json::json!({"model": "test", "api_key": "secret1"});
        let req2 = serde_json::json!({"model": "test", "api_key": "secret2"});
        assert_eq!(cache.cache_key(&req1, None), cache.cache_key(&req2, None));
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig {
            memory_max_entries: 2,
            ..Default::default()
        };
        let cache = Cache::new(config);

        let req1 = serde_json::json!({"id": 1});
        let req2 = serde_json::json!({"id": 2});
        let req3 = serde_json::json!({"id": 3});
        let val = serde_json::json!({"ok": true});

        cache.put(&req1, &val, None, true);
        cache.put(&req2, &val, None, true);
        // req1 should still be in cache
        assert!(cache.get(&req1, None).is_some());
        // Add req3 — req2 should be evicted (req1 was just accessed)
        cache.put(&req3, &val, None, true);
        assert!(cache.get(&req2, None).is_none());
        assert!(cache.get(&req1, None).is_some());
        assert!(cache.get(&req3, None).is_some());
    }

    #[test]
    fn test_disabled_cache_returns_none() {
        let config = CacheConfig {
            enable_memory_cache: false,
            enable_disk_cache: false,
            ..Default::default()
        };
        let cache = Cache::new(config);
        let request = serde_json::json!({"model": "test"});
        let response = serde_json::json!({"text": "world"});
        cache.put(&request, &response, None, true);
        assert!(cache.get(&request, None).is_none());
    }

    #[test]
    fn test_reset_memory_cache() {
        let cache = Cache::default_memory_only();
        let request = serde_json::json!({"model": "test"});
        let response = serde_json::json!({"text": "world"});
        cache.put(&request, &response, None, true);
        assert!(cache.get(&request, None).is_some());
        cache.reset_memory_cache();
        assert!(cache.get(&request, None).is_none());
    }

    #[test]
    fn test_disk_cache() {
        let temp_dir = std::env::temp_dir().join("dspy_cache_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let config = CacheConfig {
            enable_memory_cache: false,
            enable_disk_cache: true,
            disk_cache_dir: temp_dir.to_string_lossy().to_string(),
            ..Default::default()
        };
        let cache = Cache::new(config);
        let request = serde_json::json!({"model": "test", "prompt": "disk"});
        let response = serde_json::json!({"text": "from_disk"});

        cache.put(&request, &response, None, true);
        let result = cache.get(&request, None).unwrap();
        assert_eq!(result["text"], "from_disk");
        assert_eq!(result["cache_hit"], true);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_disk_promotes_to_memory() {
        let temp_dir = std::env::temp_dir().join("dspy_cache_promote_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let config = CacheConfig {
            enable_memory_cache: true,
            enable_disk_cache: true,
            disk_cache_dir: temp_dir.to_string_lossy().to_string(),
            ..Default::default()
        };
        let cache = Cache::new(config);
        let request = serde_json::json!({"model": "test", "prompt": "promote"});
        let response = serde_json::json!({"text": "promoted"});

        // Put only to disk by disabling memory
        cache.put(&request, &response, None, false);
        // Memory cache should be empty
        {
            let mem = cache.memory_cache.lock().unwrap();
            assert!(!mem.contains(&cache.cache_key(&request, None)));
        }
        // Get should promote to memory
        let result = cache.get(&request, None).unwrap();
        assert_eq!(result["text"], "promoted");
        // Now it should be in memory
        {
            let mem = cache.memory_cache.lock().unwrap();
            assert!(mem.contains(&cache.cache_key(&request, None)));
        }

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
