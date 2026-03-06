use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

pub fn docling_bin() -> PathBuf {
    if let Ok(p) = std::env::var("DOCLING_BIN") {
        return PathBuf::from(p);
    }

    let mut base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base.pop(); // e2e/ -> repo root
    base.push("docling-rs");

    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let release = PathBuf::from(&target_dir).join("release").join("docling-rs");
        if release.exists() {
            return release;
        }
        return PathBuf::from(&target_dir).join("debug").join("docling-rs");
    }

    let release = base.join("target").join("release").join("docling-rs");
    if release.exists() {
        return release;
    }
    base.join("target").join("debug").join("docling-rs")
}

pub fn test_data_dir() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.push("tests");
    path.push("data");
    path
}

pub fn groundtruth_dir() -> PathBuf {
    test_data_dir().join("groundtruth").join("docling_v2")
}

pub struct ConvertResult {
    pub output_dir: TempDir,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

pub fn run_convert(input: &Path, formats: &[&str]) -> ConvertResult {
    let tmp = TempDir::new().expect("failed to create temp dir");
    let bin = docling_bin();

    let mut cmd = Command::new(&bin);
    cmd.arg("convert")
        .arg(input)
        .arg("-o")
        .arg(tmp.path());

    for fmt in formats {
        cmd.arg("--to").arg(fmt);
    }

    let output = cmd.output().expect("failed to execute docling-rs");

    ConvertResult {
        output_dir: tmp,
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    }
}

pub fn run_convert_with_format(input: &Path, formats: &[&str], input_format: &str) -> ConvertResult {
    let tmp = TempDir::new().expect("failed to create temp dir");
    let bin = docling_bin();

    let mut cmd = Command::new(&bin);
    cmd.arg("convert")
        .arg(input)
        .arg("-o")
        .arg(tmp.path())
        .arg("--from")
        .arg(input_format);

    for fmt in formats {
        cmd.arg("--to").arg(fmt);
    }

    let output = cmd.output().expect("failed to execute docling-rs");

    ConvertResult {
        output_dir: tmp,
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    }
}

pub fn read_output(result: &ConvertResult, stem: &str, ext: &str) -> String {
    let filename = format!("{}.{}", stem, ext);
    let path = result.output_dir.path().join(&filename);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read output file {}: {}", path.display(), e))
}

pub fn read_groundtruth(name: &str) -> String {
    let path = groundtruth_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read groundtruth {}: {}", path.display(), e))
}

pub fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }

    if a.len() > 10_000 || b.len() > 10_000 {
        return line_similarity(a, b);
    }

    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein_distance(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

fn line_level_distance(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

pub fn line_similarity(a: &str, b: &str) -> f64 {
    let a_lines: Vec<&str> = a.lines().collect();
    let b_lines: Vec<&str> = b.lines().collect();
    let max_len = a_lines.len().max(b_lines.len());
    if max_len == 0 {
        return 1.0;
    }
    let dist = line_level_distance(&a_lines, &b_lines);
    1.0 - (dist as f64 / max_len as f64)
}

pub fn assert_json_structural_match(actual: &str, expected: &str) {
    let actual_val: serde_json::Value =
        serde_json::from_str(actual).expect("actual JSON is invalid");
    let expected_val: serde_json::Value =
        serde_json::from_str(expected).expect("expected JSON is invalid");

    assert_eq!(
        actual_val.get("schema_name"),
        expected_val.get("schema_name"),
        "schema_name mismatch"
    );

    let actual_tables = actual_val.get("tables").and_then(|v| v.as_array());
    let expected_tables = expected_val.get("tables").and_then(|v| v.as_array());
    if let (Some(at), Some(et)) = (actual_tables, expected_tables) {
        if !et.is_empty() {
            assert!(
                !at.is_empty(),
                "expected tables but got none (expected {})",
                et.len()
            );
        }
    }

    let actual_texts = actual_val.get("texts").and_then(|v| v.as_array());
    let expected_texts = expected_val.get("texts").and_then(|v| v.as_array());
    if let (Some(at), Some(et)) = (actual_texts, expected_texts) {
        if et.len() >= 5 {
            assert!(
                !at.is_empty(),
                "expected texts but got none (expected {})",
                et.len()
            );
        }
    }
}

pub fn assert_json_strict_structural_match(actual: &str, expected: &str) {
    let actual_val: serde_json::Value =
        serde_json::from_str(actual).expect("actual JSON is invalid");
    let expected_val: serde_json::Value =
        serde_json::from_str(expected).expect("expected JSON is invalid");

    assert_eq!(
        actual_val.get("schema_name"),
        expected_val.get("schema_name"),
        "schema_name mismatch"
    );
    assert_eq!(
        actual_val.get("name"),
        expected_val.get("name"),
        "name mismatch"
    );

    let actual_tables = actual_val.get("tables").and_then(|v| v.as_array());
    let expected_tables = expected_val.get("tables").and_then(|v| v.as_array());
    if let (Some(at), Some(et)) = (actual_tables, expected_tables) {
        assert_eq!(at.len(), et.len(), "tables count mismatch");

        for (i, (a_table, e_table)) in at.iter().zip(et.iter()).enumerate() {
            let a_data = a_table.get("data");
            let e_data = e_table.get("data");
            if let (Some(ad), Some(ed)) = (a_data, e_data) {
                assert_eq!(
                    ad.get("num_rows"),
                    ed.get("num_rows"),
                    "tables[{}].data.num_rows mismatch",
                    i
                );
                assert_eq!(
                    ad.get("num_cols"),
                    ed.get("num_cols"),
                    "tables[{}].data.num_cols mismatch",
                    i
                );

                let a_cells = ad.get("table_cells").and_then(|v| v.as_array());
                let e_cells = ed.get("table_cells").and_then(|v| v.as_array());
                if let (Some(ac), Some(ec)) = (a_cells, e_cells) {
                    assert_eq!(
                        ac.len(),
                        ec.len(),
                        "tables[{}].data.table_cells count mismatch",
                        i
                    );
                    for (j, (a_cell, e_cell)) in ac.iter().zip(ec.iter()).enumerate() {
                        assert_eq!(
                            a_cell.get("text"),
                            e_cell.get("text"),
                            "tables[{}].table_cells[{}].text mismatch",
                            i,
                            j
                        );
                        assert_eq!(
                            a_cell.get("column_header"),
                            e_cell.get("column_header"),
                            "tables[{}].table_cells[{}].column_header mismatch",
                            i,
                            j
                        );
                    }
                }
            }
        }
    }

    let actual_texts = actual_val.get("texts").and_then(|v| v.as_array());
    let expected_texts = expected_val.get("texts").and_then(|v| v.as_array());
    if let (Some(at), Some(et)) = (actual_texts, expected_texts) {
        assert_eq!(at.len(), et.len(), "texts count mismatch");
        for (i, (a_text, e_text)) in at.iter().zip(et.iter()).enumerate() {
            assert_eq!(
                a_text.get("text"),
                e_text.get("text"),
                "texts[{}].text mismatch",
                i
            );
            assert_eq!(
                a_text.get("label"),
                e_text.get("label"),
                "texts[{}].label mismatch",
                i
            );
        }
    }

    // Verify body children (check both have children, allow structural differences)
    let actual_body = actual_val.get("body").and_then(|v| v.get("children")).and_then(|v| v.as_array());
    let expected_body = expected_val.get("body").and_then(|v| v.get("children")).and_then(|v| v.as_array());
    if let (Some(ab), Some(eb)) = (actual_body, expected_body) {
        assert!(
            !ab.is_empty() || eb.is_empty(),
            "body.children: actual is empty but expected {} children",
            eb.len()
        );
    }
}

pub fn run_convert_expect_failure(input: &Path, formats: &[&str]) -> ConvertResult {
    let tmp = TempDir::new().expect("failed to create temp dir");
    let bin = docling_bin();

    let mut cmd = Command::new(&bin);
    cmd.arg("convert")
        .arg(input)
        .arg("-o")
        .arg(tmp.path())
        .arg("--abort-on-error");

    for fmt in formats {
        cmd.arg("--to").arg(fmt);
    }

    let output = cmd.output().expect("failed to execute docling-rs");

    ConvertResult {
        output_dir: tmp,
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    }
}

/// Lenient structural check: verifies the document is well-formed and produces
/// content, but allows count differences from the Python groundtruth since
/// XML/patent backends may extract different amounts of structure.
pub fn assert_json_lenient_structural_match(actual: &str, expected: &str) {
    let actual_val: serde_json::Value =
        serde_json::from_str(actual).expect("actual JSON is invalid");
    let expected_val: serde_json::Value =
        serde_json::from_str(expected).expect("expected JSON is invalid");

    assert_eq!(
        actual_val.get("schema_name"),
        expected_val.get("schema_name"),
        "schema_name mismatch"
    );

    if let (Some(an), Some(en)) = (
        actual_val.get("name").and_then(|v| v.as_str()),
        expected_val.get("name").and_then(|v| v.as_str()),
    ) {
        let an_base = an.split('.').next().unwrap_or(an);
        let en_base = en.split('.').next().unwrap_or(en);
        assert_eq!(an_base, en_base, "name base mismatch: actual={}, expected={}", an, en);
    }

    let actual_texts = actual_val.get("texts").and_then(|v| v.as_array());
    let expected_texts = expected_val.get("texts").and_then(|v| v.as_array());
    if let (Some(at), Some(et)) = (actual_texts, expected_texts) {
        if !et.is_empty() {
            assert!(!at.is_empty(), "expected texts but got none");
            let ratio = at.len() as f64 / et.len() as f64;
            assert!(
                ratio >= 0.3,
                "texts count too low: actual={}, expected={}, ratio={:.2}",
                at.len(),
                et.len(),
                ratio
            );
        }
    }
}

/// Validates that the output JSON is a structurally valid DoclingDocument:
/// correct schema_name, body with children, and all $ref pointers resolve.
pub fn assert_valid_docling_document(json: &str) {
    let val: serde_json::Value = serde_json::from_str(json).expect("output should be valid JSON");

    assert_eq!(
        val.get("schema_name").and_then(|v| v.as_str()),
        Some("DoclingDocument"),
        "schema_name must be 'DoclingDocument'"
    );

    assert!(val.get("name").is_some(), "document must have a name");
    assert!(val.get("origin").is_some(), "document must have an origin");

    let body = val.get("body").expect("document must have a body");
    let body_children = body.get("children").and_then(|v| v.as_array());
    assert!(body_children.is_some(), "body must have children array");

    let texts = val.get("texts").and_then(|v| v.as_array());
    let tables = val.get("tables").and_then(|v| v.as_array());
    let groups = val.get("groups").and_then(|v| v.as_array());
    let pictures = val.get("pictures").and_then(|v| v.as_array());

    let texts_len = texts.map(|t| t.len()).unwrap_or(0);
    let tables_len = tables.map(|t| t.len()).unwrap_or(0);
    let groups_len = groups.map(|g| g.len()).unwrap_or(0);
    let pictures_len = pictures.map(|p| p.len()).unwrap_or(0);

    for child_ref in body_children.unwrap() {
        let ref_str = child_ref
            .get("$ref")
            .and_then(|v| v.as_str())
            .expect("body child must have a $ref");
        let valid = if let Some(rest) = ref_str.strip_prefix("#/texts/") {
            rest.parse::<usize>().map(|i| i < texts_len).unwrap_or(false)
        } else if let Some(rest) = ref_str.strip_prefix("#/tables/") {
            rest.parse::<usize>().map(|i| i < tables_len).unwrap_or(false)
        } else if let Some(rest) = ref_str.strip_prefix("#/groups/") {
            rest.parse::<usize>().map(|i| i < groups_len).unwrap_or(false)
        } else if let Some(rest) = ref_str.strip_prefix("#/pictures/") {
            rest.parse::<usize>().map(|i| i < pictures_len).unwrap_or(false)
        } else {
            false
        };
        assert!(
            valid,
            "body child $ref '{}' does not resolve to a valid entry",
            ref_str
        );
    }
}

/// Normalize Python's `cref` keys to `$ref` for comparison with Rust output.
fn normalize_refs(val: &serde_json::Value) -> serde_json::Value {
    match val {
        serde_json::Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (k, v) in map {
                let key = if k == "cref" { "$ref".to_string() } else { k.clone() };
                new_map.insert(key, normalize_refs(v));
            }
            serde_json::Value::Object(new_map)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(normalize_refs).collect())
        }
        other => other.clone(),
    }
}

fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

pub fn assert_md_similar(actual: &str, expected: &str, min_similarity: f64) {
    let sim = levenshtein_similarity(actual.trim(), expected.trim());
    assert!(
        sim >= min_similarity,
        "Markdown similarity {:.2}% is below threshold {:.2}%\n--- Actual ---\n{}\n--- Expected ---\n{}",
        sim * 100.0,
        min_similarity * 100.0,
        safe_truncate(actual, 500),
        safe_truncate(expected, 500),
    );
}
