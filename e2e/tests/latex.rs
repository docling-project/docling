use docling_e2e::helpers::*;

fn latex_test_file(input: &std::path::Path, filename: &str) {
    let stem = filename.strip_suffix(".tex").unwrap_or(filename);

    let result = run_convert(input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, stem, "json");
    let gt_json_name = format!("{}.json", filename);
    let expected_json = read_groundtruth(&gt_json_name);
    assert_json_structural_match(&actual_json, &expected_json);

    let actual_md = read_output(&result, stem, "md");
    let gt_md_name = format!("{}.md", filename);
    let expected_md = read_groundtruth(&gt_md_name);
    assert_md_similar(&actual_md, &expected_md, 0.70);
}

#[test]
fn test_latex_example_01() {
    let input = test_data_dir().join("latex").join("example_01.tex");
    latex_test_file(&input, "example_01.tex");
}

#[test]
fn test_latex_example_02() {
    let input = test_data_dir().join("latex").join("example_02.tex");
    latex_test_file(&input, "example_02.tex");
}

#[test]
fn test_latex_multifile_1706() {
    let input = test_data_dir().join("latex").join("1706.03762").join("main.tex");

    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "main", "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");

    let texts = parsed.get("texts").and_then(|v| v.as_array());
    assert!(
        texts.map_or(false, |t| t.len() > 5),
        "multi-file LaTeX should produce texts from \\input includes"
    );

    let actual_md = read_output(&result, "main", "md");
    assert!(
        actual_md.len() > 500,
        "multi-file LaTeX should produce substantial markdown"
    );

    assert!(
        actual_md.contains("Introduction")
            || actual_md.contains("Background")
            || actual_md.contains("Attention"),
        "markdown should contain section content from included files"
    );
}

#[test]
fn test_latex_2310_06825_smoke() {
    let input = test_data_dir().join("latex").join("2310.06825").join("main.tex");
    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "main", "json");
    let _parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
}

#[test]
fn test_latex_2305_03393_smoke() {
    let input = test_data_dir().join("latex").join("2305.03393").join("main.tex");
    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "main", "json");
    let _parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
}

#[test]
fn test_latex_2501_00089_smoke() {
    let input = test_data_dir().join("latex").join("2501.00089").join("main.tex");
    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "main", "json");
    let _parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
}

#[test]
fn test_latex_2412_19437_smoke() {
    let input = test_data_dir()
        .join("latex")
        .join("2412.19437")
        .join("main.tex");
    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "main", "json");
    let _parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
}

#[test]
fn test_latex_arxiv_2501_01300v2_smoke() {
    let input = test_data_dir()
        .join("latex")
        .join("arXiv-2501.01300v2")
        .join("main.tex");
    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "main", "json");
    let _parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
}

#[test]
fn test_latex_example_01_text_count() {
    let input = test_data_dir().join("latex").join("example_01.tex");

    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "example_01", "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");

    let expected_json = read_groundtruth("example_01.tex.json");
    let expected: serde_json::Value =
        serde_json::from_str(&expected_json).expect("groundtruth should be valid JSON");

    let actual_texts = parsed.get("texts").and_then(|v| v.as_array()).unwrap();
    let expected_texts = expected.get("texts").and_then(|v| v.as_array()).unwrap();

    assert_eq!(
        actual_texts.len(),
        expected_texts.len(),
        "text item count should match groundtruth"
    );

    let actual_groups = parsed.get("groups").and_then(|v| v.as_array()).unwrap();
    let expected_groups = expected.get("groups").and_then(|v| v.as_array()).unwrap();
    assert_eq!(
        actual_groups.len(),
        expected_groups.len(),
        "group count should match groundtruth"
    );
}

#[test]
fn test_latex_example_02_has_formula() {
    let input = test_data_dir().join("latex").join("example_02.tex");

    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "example_02", "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");

    let texts = parsed.get("texts").and_then(|v| v.as_array()).unwrap();

    let has_formula = texts.iter().any(|t| t.get("label").and_then(|l| l.as_str()) == Some("formula"));
    assert!(has_formula, "example_02 should have at least one formula item");

    let tables = parsed.get("tables").and_then(|v| v.as_array()).unwrap();
    assert_eq!(tables.len(), 1, "example_02 should have exactly one table");
}

#[test]
fn test_latex_example_02_citation_format() {
    let input = test_data_dir().join("latex").join("example_02.tex");

    let result = run_convert(&input, &["md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_md = read_output(&result, "example_02", "md");
    assert!(
        actual_md.contains("[smith2020]"),
        "\\cite{{smith2020}} should render as [smith2020] in markdown, got:\n{}",
        &actual_md[actual_md.len().saturating_sub(200)..]
    );
    assert!(
        actual_md.contains("[sec:math]"),
        "\\ref{{sec:math}} should render as [sec:math] in markdown"
    );
}
