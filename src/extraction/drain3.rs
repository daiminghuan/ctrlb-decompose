use std::collections::HashMap;
use std::fmt;
use std::num::NonZero;

use lru::LruCache;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::types::{PatternID, VarType};

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    max_node_depth: usize,
    log_cluster_depth: usize,
    sim_th: f64,
    max_children: usize,
    extra_delimiters: Vec<String>,
    max_clusters: usize,
    param_string: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_node_depth: 0, // Will be set to log_cluster_depth - 2 in Drain::new
            log_cluster_depth: 4,
            sim_th: 0.4,
            max_children: 100,
            extra_delimiters: Vec::new(),
            max_clusters: 10000, // 0 means no limit
            param_string: "<*>".to_string(),
        }
    }
}

/// A typed variable extracted from a log line
#[derive(Debug, Clone, PartialEq)]
pub struct TypedVariable {
    pub raw: String,
    pub var_type: VarType,
}

/// Result of parsing a log line through Drain3
pub struct ParsedLog {
    pub pattern_id: PatternID,
    pub template: String,
    pub count: u64,
    pub variables: Vec<TypedVariable>,
}

// --- Variable type classification ---

static RE_UUID: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
        .unwrap()
});

static RE_IPV4: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?$").unwrap()
});

static RE_IPV6: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$").unwrap()
});

static RE_DURATION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^-?\d+\.?\d*(ns|us|µs|ms|s|m|h)$").unwrap()
});

static RE_ISO_TIMESTAMP: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}").unwrap()
});

static RE_HEX_ID: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(0x)?[0-9a-fA-F]{4,}$").unwrap()
});

/// Classify a raw variable string into a VarType.
/// Uses fast character checks first, falling back to regex only when needed.
pub fn classify_variable(value: &str) -> VarType {
    let bytes = value.as_bytes();
    let len = bytes.len();

    if len == 0 {
        return VarType::String;
    }

    let first = bytes[0];

    // UUID: exactly 36 chars, pattern xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    if len == 36 && bytes[8] == b'-' && bytes[13] == b'-' && bytes[18] == b'-' && bytes[23] == b'-' {
        if value.bytes().enumerate().all(|(i, b)| {
            if i == 8 || i == 13 || i == 18 || i == 23 { b == b'-' }
            else { b.is_ascii_hexdigit() }
        }) {
            return VarType::UUID;
        }
    }

    // Fast path: starts with digit or '-' followed by digit
    let starts_numeric = first.is_ascii_digit() || (first == b'-' && len > 1 && bytes[1].is_ascii_digit());

    if starts_numeric {
        // Duration: ends with time unit suffix
        if len >= 2 {
            let last = bytes[len - 1];
            let is_duration = match last {
                b's' => {
                    if len >= 3 {
                        match bytes[len - 2] {
                            b'n' | b'm' | b'u' => true, // ns, ms, us
                            _ => bytes[len - 2].is_ascii_digit(), // plain 's'
                        }
                    } else {
                        true
                    }
                }
                b'm' | b'h' => bytes[len - 2].is_ascii_digit(),
                _ => false,
            };
            if is_duration && RE_DURATION.is_match(value) {
                return VarType::Duration;
            }
        }

        // Check for dots — could be IPv4 or Float
        if value.contains('.') {
            // IPv4: digit.digit.digit.digit with optional :port
            let ip_part = value.split(':').next().unwrap_or(value);
            let dot_count = ip_part.bytes().filter(|&b| b == b'.').count();
            if dot_count == 3 && RE_IPV4.is_match(value) {
                let valid = ip_part
                    .split('.')
                    .all(|octet| octet.parse::<u16>().is_ok_and(|n| n <= 255));
                if valid {
                    return VarType::IPv4;
                }
            }

            // Float
            let check = if first == b'-' { &value[1..] } else { value };
            if check.bytes().all(|b| b.is_ascii_digit() || b == b'.') && check.parse::<f64>().is_ok() {
                return VarType::Float;
            }
        }

        // ISO timestamp: starts with 4 digits then '-'
        if len >= 19 && bytes[4] == b'-' && bytes[7] == b'-' {
            return VarType::Timestamp;
        }

        // Integer
        if value.parse::<i64>().is_ok() {
            return VarType::Integer;
        }
    }

    // IPv6: contains multiple ':'
    if value.contains(':') && value.bytes().filter(|&b| b == b':').count() >= 2 {
        if RE_IPV6.is_match(value) {
            return VarType::IPv6;
        }
    }

    // HexID: starts with 0x or is all hex with at least one a-f
    if len >= 4 {
        let hex_part = if bytes[0] == b'0' && bytes[1] == b'x' { &value[2..] } else { value };
        if hex_part.len() >= 4
            && hex_part.bytes().all(|b| b.is_ascii_hexdigit())
            && hex_part.bytes().any(|b| b.is_ascii_hexdigit() && !b.is_ascii_digit())
        {
            return VarType::HexID;
        }
    }

    // µs duration (multi-byte prefix)
    if value.ends_with("µs") && RE_DURATION.is_match(value) {
        return VarType::Duration;
    }

    VarType::String
}

pub struct LogCluster {
    log_template_tokens: Vec<String>,
    id: usize,
    size: usize,
}

impl LogCluster {
    fn get_template(&self) -> String {
        self.log_template_tokens.join(" ")
    }
}

impl fmt::Display for LogCluster {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "id={{{}}} : size={{{}}} : {}",
            self.id,
            self.size,
            self.get_template()
        )
    }
}

struct LogClusterCache {
    cache: LruCache<usize, LogCluster>,
}

impl LogClusterCache {
    fn new(max_size: usize) -> Self {
        // 0 means no limit — use unbounded LRU
        let size = if max_size == 0 {
            NonZero::new(usize::MAX).unwrap()
        } else {
            NonZero::new(max_size).unwrap()
        };

        LogClusterCache {
            cache: LruCache::new(size),
        }
    }

    fn values(&mut self) -> Vec<LogCluster> {
        let mut values = Vec::new();

        // Create a list of keys to avoid borrowing issues
        let keys: Vec<usize> = self.cache.iter().map(|(k, _)| *k).collect();

        for key in keys {
            if let Some(cluster) = self.cache.get(&key) {
                values.push(LogCluster {
                    log_template_tokens: cluster.log_template_tokens.clone(),
                    id: cluster.id,
                    size: cluster.size,
                });
            }
        }

        values
    }

    fn set(&mut self, key: usize, cluster: LogCluster) {
        self.cache.put(key, cluster);
    }

    fn get(&mut self, key: usize) -> Option<&mut LogCluster> {
        self.cache.get_mut(&key)
    }

    fn peek(&mut self, key: usize) -> Option<&LogCluster> {
        self.cache.peek(&key)
    }
}

#[derive(Debug, Clone)]
struct Node {
    key_to_child_node: HashMap<String, Node>,
    cluster_ids: Vec<usize>,
}

impl Node {
    fn new() -> Self {
        Node {
            key_to_child_node: HashMap::new(),
            cluster_ids: Vec::new(),
        }
    }
}

pub struct Drain {
    config: Config,
    root_node: Node,
    id_to_cluster: LogClusterCache,
    clusters_counter: usize,
}

impl Drain {
    pub fn new(config: Config) -> Self {
        let mut local_config = config.clone();
        if config.log_cluster_depth < 3 {
            panic!("depth argument must be at least 3");
        }
        local_config.max_node_depth = config.log_cluster_depth - 2;

        Drain {
            config: local_config.clone(),
            root_node: Node::new(),
            id_to_cluster: LogClusterCache::new(local_config.max_clusters),
            clusters_counter: 0,
        }
    }

    pub fn clusters(&mut self) -> Vec<LogCluster> {
        self.id_to_cluster.values()
    }

    pub fn train(&mut self, content: &str) -> LogCluster {
        let content_tokens = self.get_content_as_tokens(content);
        let (id, size) = self.train_with_tokens(&content_tokens);
        if let Some(cluster) = self.id_to_cluster.peek(id) {
            LogCluster {
                log_template_tokens: cluster.log_template_tokens.clone(),
                id: cluster.id,
                size: cluster.size,
            }
        } else {
            LogCluster {
                log_template_tokens: Vec::new(),
                id,
                size,
            }
        }
    }

    fn train_with_tokens(&mut self, content_tokens: &[String]) -> (usize, usize) {
        let sim_th = self.config.sim_th;
        let match_id = self.tree_search_impl_fast(content_tokens, sim_th, false);

        if let Some(matched_id) = match_id {
            // Update template in-place — only clone param_string when a token actually differs
            if let Some(cluster) = self.id_to_cluster.get(matched_id) {
                let template_len = cluster.log_template_tokens.len();
                let check_len = content_tokens.len().min(template_len);
                for i in 0..check_len {
                    if cluster.log_template_tokens[i] != self.config.param_string
                        && content_tokens[i] != cluster.log_template_tokens[i]
                    {
                        cluster.log_template_tokens[i].clear();
                        cluster.log_template_tokens[i].push_str(&self.config.param_string);
                    }
                }
                cluster.size += 1;
                (matched_id, cluster.size)
            } else {
                panic!("Cluster not found in cache after match");
            }
        } else {
            self.clusters_counter += 1;
            let cluster_id = self.clusters_counter;
            self.add_seq_to_prefix_tree_helper(cluster_id, content_tokens);
            let cluster = LogCluster {
                log_template_tokens: content_tokens.to_vec(),
                id: cluster_id,
                size: 1,
            };
            self.id_to_cluster.set(cluster_id, cluster);
            (cluster_id, 1)
        }
    }

    // Match against an already existing cluster. Match shall be perfect (sim_th=1.0).
    // New cluster will not be created as a result of this call, nor any cluster modifications.
    pub fn match_log(&mut self, content: &str) -> Option<LogCluster> {
        let content_tokens = self.get_content_as_tokens(content);
        let root_node = self.root_node.clone();
        self.tree_search(&root_node, &content_tokens, 1.0, true)
    }

    fn get_content_as_tokens(&self, content: &str) -> Vec<String> {
        if self.config.extra_delimiters.is_empty() {
            content.split_whitespace().map(|s| s.to_string()).collect()
        } else {
            let mut content = content.trim().to_string();
            for extra_delimiter in &self.config.extra_delimiters {
                content = content.replace(extra_delimiter, " ");
            }
            content.split_whitespace().map(|s| s.to_string()).collect()
        }
    }

    pub fn extract_template_and_vars(&mut self, content: &str) -> ParsedLog {
        let content_tokens = self.get_content_as_tokens(content);

        // Train (one tokenization only) and get cluster id
        let (cluster_id, cluster_size) = self.train_with_tokens(&content_tokens);

        // Extract variables by comparing the template with the original content
        let mut variables = Vec::new();
        let template = if let Some(cluster) = self.id_to_cluster.peek(cluster_id) {
            let param_str = &self.config.param_string;
            for (i, token) in content_tokens.iter().enumerate() {
                if i < cluster.log_template_tokens.len()
                    && cluster.log_template_tokens[i] == *param_str
                {
                    variables.push(TypedVariable {
                        var_type: classify_variable(token),
                        raw: token.clone(),
                    });
                }
            }
            cluster.log_template_tokens.join(" ")
        } else {
            String::new()
        };

        ParsedLog {
            pattern_id: cluster_id,
            template,
            count: cluster_size as u64,
            variables,
        }
    }

    fn tree_search_impl_fast(
        &mut self,
        tokens: &[String],
        sim_th: f64,
        include_params: bool,
    ) -> Option<usize> {
        let token_count = tokens.len();
        let token_count_str = token_count.to_string();

        if !self
            .root_node
            .key_to_child_node
            .contains_key(&token_count_str)
        {
            return None;
        }

        let cluster_ids: Vec<usize>;
        {
            let cur_node = &self.root_node.key_to_child_node[&token_count_str];

            if token_count == 0 {
                if !cur_node.cluster_ids.is_empty() {
                    cluster_ids = cur_node.cluster_ids.clone();
                } else {
                    return None;
                }
            } else {
                let mut cur_node = cur_node;
                let mut cur_node_depth = 1;
                let param_str = &self.config.param_string;
                let max_depth = self.config.max_node_depth;

                for i in 0..tokens.len() {
                    if cur_node_depth >= max_depth {
                        break;
                    }
                    if cur_node_depth == token_count {
                        break;
                    }

                    let token = &tokens[i];
                    if cur_node.key_to_child_node.contains_key(token) {
                        cur_node = &cur_node.key_to_child_node[token];
                    } else if cur_node.key_to_child_node.contains_key(param_str) {
                        cur_node = &cur_node.key_to_child_node[param_str];
                    } else {
                        return None;
                    }

                    cur_node_depth += 1;
                }

                cluster_ids = cur_node.cluster_ids.clone();
            }
        }

        self.fast_match_id(&cluster_ids, tokens, sim_th, include_params)
    }

    fn tree_search_impl(
        &mut self,
        tokens: &[String],
        sim_th: f64,
        include_params: bool,
    ) -> Option<LogCluster> {
        if let Some(id) = self.tree_search_impl_fast(tokens, sim_th, include_params) {
            self.id_to_cluster.peek(id).map(|c| LogCluster {
                log_template_tokens: c.log_template_tokens.clone(),
                id: c.id,
                size: c.size,
            })
        } else {
            None
        }
    }

    // Helper method to update the prefix tree while avoiding borrow issues
    fn add_seq_to_prefix_tree_helper(&mut self, cluster_id: usize, tokens: &[String]) {
        // Store values needed by add_seq_to_prefix_tree to avoid borrowing self completely
        let max_node_depth = self.config.max_node_depth;
        let param_string = self.config.param_string.clone();
        let max_children = self.config.max_children;

        // Update the tree structure
        add_seq_to_prefix_tree_impl(
            &mut self.root_node,
            cluster_id,
            tokens,
            max_node_depth,
            &param_string,
            max_children,
            &mut self.id_to_cluster,
        );
    }

    fn tree_search(
        &mut self,
        root_node: &Node,
        tokens: &[String],
        sim_th: f64,
        include_params: bool,
    ) -> Option<LogCluster> {
        let token_count = tokens.len();
        let token_count_str = token_count.to_string();

        // at first level, children are grouped by token (word) count
        if !root_node.key_to_child_node.contains_key(&token_count_str) {
            return None;
        }

        let cur_node = &root_node.key_to_child_node[&token_count_str];

        // handle case of empty log string - return the single cluster in that group
        if token_count == 0 {
            if !cur_node.cluster_ids.is_empty() {
                if let Some(cluster) = self.id_to_cluster.get(cur_node.cluster_ids[0]) {
                    return Some(LogCluster {
                        log_template_tokens: cluster.log_template_tokens.clone(),
                        id: cluster.id,
                        size: cluster.size,
                    });
                }
            }
            return None;
        }

        // find the leaf node for this log - a path of nodes matching the first N tokens (N=tree depth)
        let mut cur_node = cur_node;
        let mut cur_node_depth = 1;

        for i in 0..tokens.len() {
            // at max depth
            if cur_node_depth >= self.config.max_node_depth {
                break;
            }

            // this is last token
            if cur_node_depth == token_count {
                break;
            }

            let token = &tokens[i];
            let param_str = &self.config.param_string;

            if cur_node.key_to_child_node.contains_key(token) {
                cur_node = &cur_node.key_to_child_node[token];
            } else if cur_node.key_to_child_node.contains_key(param_str) {
                cur_node = &cur_node.key_to_child_node[param_str];
            } else {
                return None;
            }

            cur_node_depth += 1;
        }

        // get best match among all clusters with same prefix, or None if no match is above sim_th
        self.fast_match(&cur_node.cluster_ids, tokens, sim_th, include_params)
    }

    fn fast_match_id(
        &mut self,
        cluster_ids: &[usize],
        tokens: &[String],
        sim_th: f64,
        include_params: bool,
    ) -> Option<usize> {
        let mut max_sim: f64 = -1.0;
        let mut max_param_count: i32 = -1;
        let mut best_id: Option<usize> = None;

        let param_string = &self.config.param_string as *const String;

        for &cluster_id in cluster_ids {
            if let Some(cluster) = self.id_to_cluster.peek(cluster_id) {
                // SAFETY: param_string points to self.config.param_string which is not moved
                let (cur_sim, param_count) = get_seq_distance_static(
                    &cluster.log_template_tokens,
                    tokens,
                    include_params,
                    unsafe { &*param_string },
                );

                if cur_sim > max_sim || (cur_sim == max_sim && param_count > max_param_count) {
                    max_sim = cur_sim;
                    max_param_count = param_count;
                    best_id = Some(cluster_id);
                }
            }
        }

        if max_sim >= sim_th {
            best_id
        } else {
            None
        }
    }

    // fastMatch Find the best match for a log message (represented as tokens) versus a list of clusters
    fn fast_match(
        &mut self,
        cluster_ids: &[usize],
        tokens: &[String],
        sim_th: f64,
        include_params: bool,
    ) -> Option<LogCluster> {
        if let Some(id) = self.fast_match_id(cluster_ids, tokens, sim_th, include_params) {
            self.id_to_cluster.peek(id).map(|c| LogCluster {
                log_template_tokens: c.log_template_tokens.clone(),
                id: c.id,
                size: c.size,
            })
        } else {
            None
        }
    }

    fn get_seq_distance(
        &self,
        seq1: &[String],
        seq2: &[String],
        include_params: bool,
    ) -> (f64, i32) {
        get_seq_distance_static(seq1, seq2, include_params, &self.config.param_string)
    }

    fn add_seq_to_prefix_tree(
        &mut self,
        root_node: &mut Node,
        cluster_id: usize,
        tokens: &[String],
    ) {
        let token_count = tokens.len();
        let token_count_str = token_count.to_string();

        if !root_node.key_to_child_node.contains_key(&token_count_str) {
            root_node
                .key_to_child_node
                .insert(token_count_str.clone(), Node::new());
        }

        let first_layer_node = root_node
            .key_to_child_node
            .get_mut(&token_count_str)
            .unwrap();

        // handle case of empty log string
        if token_count == 0 {
            first_layer_node.cluster_ids.push(cluster_id);
            return;
        }

        let mut cur_node = first_layer_node;
        let mut current_depth = 1;

        for i in 0..tokens.len() {
            let token = &tokens[i];

            // if at max depth or this is last token in template - add current log cluster to the leaf node
            if current_depth >= self.config.max_node_depth || current_depth >= token_count {
                // Clean up stale clusters before adding a new one
                let mut new_cluster_ids = Vec::new();

                for &id in &cur_node.cluster_ids {
                    if self.id_to_cluster.peek(id).is_some() {
                        new_cluster_ids.push(id);
                    }
                }

                new_cluster_ids.push(cluster_id);
                cur_node.cluster_ids = new_cluster_ids;
                break;
            }

            // if token not matched in this layer of existing tree
            if !cur_node.key_to_child_node.contains_key(token) {
                // if token doesn't contain any numbers
                if !Self::has_numbers(token) {
                    let param_str = self.config.param_string.clone();

                    if cur_node.key_to_child_node.contains_key(&param_str) {
                        if cur_node.key_to_child_node.len() < self.config.max_children {
                            cur_node
                                .key_to_child_node
                                .insert(token.clone(), Node::new());
                            cur_node = cur_node.key_to_child_node.get_mut(token).unwrap();
                        } else {
                            cur_node = cur_node.key_to_child_node.get_mut(&param_str).unwrap();
                        }
                    } else if cur_node.key_to_child_node.len() + 1 < self.config.max_children {
                        cur_node
                            .key_to_child_node
                            .insert(token.clone(), Node::new());
                        cur_node = cur_node.key_to_child_node.get_mut(token).unwrap();
                    } else if cur_node.key_to_child_node.len() + 1 == self.config.max_children {
                        cur_node
                            .key_to_child_node
                            .insert(param_str.clone(), Node::new());
                        cur_node = cur_node.key_to_child_node.get_mut(&param_str).unwrap();
                    } else {
                        cur_node = cur_node.key_to_child_node.get_mut(&param_str).unwrap();
                    }
                } else {
                    let param_str = self.config.param_string.clone();

                    if !cur_node.key_to_child_node.contains_key(&param_str) {
                        cur_node
                            .key_to_child_node
                            .insert(param_str.clone(), Node::new());
                    }

                    cur_node = cur_node.key_to_child_node.get_mut(&param_str).unwrap();
                }
            } else {
                // if the token is matched
                cur_node = cur_node.key_to_child_node.get_mut(token).unwrap();
            }

            current_depth += 1;
        }
    }

    fn has_numbers(s: &str) -> bool {
        s.chars().any(|c| c.is_numeric())
    }

    fn create_template(&self, seq1: &[String], seq2: &[String]) -> Vec<String> {
        if seq1.len() != seq2.len() {
            panic!("seq1 and seq2 must be of same length");
        }

        let mut ret_val = seq2.to_vec();

        for i in 0..seq1.len() {
            if seq1[i] != seq2[i] {
                ret_val[i] = self.config.param_string.clone();
            }
        }

        ret_val
    }
}

fn get_seq_distance_static(
    seq1: &[String],
    seq2: &[String],
    include_params: bool,
    param_string: &str,
) -> (f64, i32) {
    if seq1.len() != seq2.len() {
        panic!("seq1 and seq2 must be of same length");
    }

    let mut sim_tokens = 0;
    let mut param_count = 0;

    for i in 0..seq1.len() {
        let token1 = &seq1[i];
        let token2 = &seq2[i];

        if token1 == param_string {
            param_count += 1;
        } else if token1 == token2 {
            sim_tokens += 1;
        }
    }

    if include_params {
        sim_tokens += param_count;
    }

    let ret_val = sim_tokens as f64 / seq1.len() as f64;
    (ret_val, param_count)
}

// Static implementation of add_seq_to_prefix_tree to avoid borrowing issues
fn add_seq_to_prefix_tree_impl(
    root_node: &mut Node,
    cluster_id: usize,
    tokens: &[String],
    max_node_depth: usize,
    param_string: &str,
    max_children: usize,
    id_to_cluster: &mut LogClusterCache,
) {
    let token_count = tokens.len();
    let token_count_str = token_count.to_string();

    if !root_node.key_to_child_node.contains_key(&token_count_str) {
        root_node
            .key_to_child_node
            .insert(token_count_str.clone(), Node::new());
    }

    let first_layer_node = root_node
        .key_to_child_node
        .get_mut(&token_count_str)
        .unwrap();

    // handle case of empty log string
    if token_count == 0 {
        first_layer_node.cluster_ids.push(cluster_id);
        return;
    }

    let mut cur_node = first_layer_node;
    let mut current_depth = 1;

    for i in 0..tokens.len() {
        let token = &tokens[i];

        // if at max depth or this is last token in template - add current log cluster to the leaf node
        if current_depth >= max_node_depth || current_depth >= token_count {
            // Clean up stale clusters before adding a new one
            let mut new_cluster_ids = Vec::new();

            for &id in &cur_node.cluster_ids {
                if id_to_cluster.peek(id).is_some() {
                    new_cluster_ids.push(id);
                }
            }

            new_cluster_ids.push(cluster_id);
            cur_node.cluster_ids = new_cluster_ids;
            break;
        }

        // if token not matched in this layer of existing tree
        if !cur_node.key_to_child_node.contains_key(token) {
            // if token doesn't contain any numbers
            if !has_numbers(token) {
                if cur_node.key_to_child_node.contains_key(param_string) {
                    if cur_node.key_to_child_node.len() < max_children {
                        cur_node
                            .key_to_child_node
                            .insert(token.clone(), Node::new());
                        cur_node = cur_node.key_to_child_node.get_mut(token).unwrap();
                    } else {
                        cur_node = cur_node.key_to_child_node.get_mut(param_string).unwrap();
                    }
                } else if cur_node.key_to_child_node.len() + 1 < max_children {
                    cur_node
                        .key_to_child_node
                        .insert(token.clone(), Node::new());
                    cur_node = cur_node.key_to_child_node.get_mut(token).unwrap();
                } else if cur_node.key_to_child_node.len() + 1 == max_children {
                    cur_node
                        .key_to_child_node
                        .insert(param_string.to_string(), Node::new());
                    cur_node = cur_node.key_to_child_node.get_mut(param_string).unwrap();
                } else {
                    cur_node = cur_node.key_to_child_node.get_mut(param_string).unwrap();
                }
            } else {
                if !cur_node.key_to_child_node.contains_key(param_string) {
                    cur_node
                        .key_to_child_node
                        .insert(param_string.to_string(), Node::new());
                }

                cur_node = cur_node.key_to_child_node.get_mut(param_string).unwrap();
            }
        } else {
            // if the token is matched
            cur_node = cur_node.key_to_child_node.get_mut(token).unwrap();
        }

        current_depth += 1;
    }
}

// Standalone function to check if a string contains numbers
fn has_numbers(s: &str) -> bool {
    s.chars().any(|c| c.is_numeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drain3() {
        let config = Config::default();
        let mut drain = Drain::new(config);

        drain.train("User logged in successfully");
        drain.train("User logged in successfully");
        drain.train("User authentication failed");
        drain.train("User logged in not successfully");

        println!("Log clusters:");
        for cluster in drain.clusters() {
            println!("{}", cluster);
        }

        let log = "User ID 124 Logged in Successfully";
        let parsed = drain.extract_template_and_vars(log);
        println!("Template: {}", parsed.template);
        println!("Variables: {:?}", parsed.variables);
        assert!(parsed.pattern_id > 0);
        assert!(parsed.count >= 1);

        // 4 clusters: the 3 original + "User ID 124 Logged in Successfully" (6 tokens, no match)
        assert_eq!(drain.clusters().len(), 4);
    }

    #[test]
    fn test_classify_integer() {
        assert_eq!(classify_variable("42"), VarType::Integer);
        assert_eq!(classify_variable("-100"), VarType::Integer);
        assert_eq!(classify_variable("0"), VarType::Integer);
    }

    #[test]
    fn test_classify_float() {
        assert_eq!(classify_variable("3.14"), VarType::Float);
        assert_eq!(classify_variable("-0.5"), VarType::Float);
        assert_eq!(classify_variable("100.0"), VarType::Float);
    }

    #[test]
    fn test_classify_duration() {
        assert_eq!(classify_variable("45ms"), VarType::Duration);
        assert_eq!(classify_variable("1.2s"), VarType::Duration);
        assert_eq!(classify_variable("500us"), VarType::Duration);
        assert_eq!(classify_variable("100ns"), VarType::Duration);
        assert_eq!(classify_variable("2h"), VarType::Duration);
    }

    #[test]
    fn test_classify_ipv4() {
        assert_eq!(classify_variable("10.0.1.15"), VarType::IPv4);
        assert_eq!(classify_variable("192.168.1.1:8080"), VarType::IPv4);
        assert_eq!(classify_variable("255.255.255.255"), VarType::IPv4);
        // Invalid octets should fall through
        assert_ne!(classify_variable("999.999.999.999"), VarType::IPv4);
    }

    #[test]
    fn test_classify_ipv6() {
        assert_eq!(classify_variable("::1"), VarType::IPv6);
        assert_eq!(classify_variable("fe80::1"), VarType::IPv6);
        assert_eq!(
            classify_variable("2001:0db8:85a3:0000:0000:8a2e:0370:7334"),
            VarType::IPv6
        );
    }

    #[test]
    fn test_classify_uuid() {
        assert_eq!(
            classify_variable("550e8400-e29b-41d4-a716-446655440000"),
            VarType::UUID
        );
        assert_eq!(
            classify_variable("123E4567-E89B-12D3-A456-426614174000"),
            VarType::UUID
        );
    }

    #[test]
    fn test_classify_hex_id() {
        assert_eq!(classify_variable("0x1a2b3c"), VarType::HexID);
        assert_eq!(classify_variable("deadbeef"), VarType::HexID);
        assert_eq!(classify_variable("abc123def456"), VarType::HexID);
        // Pure digits should be Integer, not HexID
        assert_eq!(classify_variable("1234"), VarType::Integer);
    }

    #[test]
    fn test_classify_timestamp() {
        assert_eq!(
            classify_variable("2024-01-15T14:22:01.123Z"),
            VarType::Timestamp
        );
        assert_eq!(
            classify_variable("2024-01-15 14:22:01"),
            VarType::Timestamp
        );
    }

    #[test]
    fn test_classify_string_fallback() {
        assert_eq!(classify_variable("GET"), VarType::String);
        assert_eq!(classify_variable("/api/users"), VarType::String);
        assert_eq!(classify_variable("some-random-text"), VarType::String);
    }

    #[test]
    fn test_extract_typed_variables() {
        let config = Config::default();
        let mut drain = Drain::new(config);

        // Train with similar lines to establish a pattern
        drain.train("Request from 10.0.1.15 completed in 45ms status=200");
        let parsed =
            drain.extract_template_and_vars("Request from 192.168.1.1 completed in 100ms status=500");

        // Should have variables extracted with types
        assert!(!parsed.variables.is_empty());
        // Check that IP is classified correctly
        let ip_var = parsed.variables.iter().find(|v| v.var_type == VarType::IPv4);
        assert!(ip_var.is_some());
    }
}