use docling_e2e::helpers::*;

fn csv_test(filename: &str) {
    let input = test_data_dir().join("csv").join(filename);
    let stem = filename.strip_suffix(".csv").unwrap_or(filename);

    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    // JSON structural check
    let actual_json = read_output(&result, stem, "json");
    let gt_json_name = format!("{}.json", filename);
    let expected_json = read_groundtruth(&gt_json_name);
    assert_json_strict_structural_match(&actual_json, &expected_json);

    // Markdown similarity check
    let actual_md = read_output(&result, stem, "md");
    let gt_md_name = format!("{}.md", filename);
    let expected_md = read_groundtruth(&gt_md_name);
    assert_md_similar(&actual_md, &expected_md, 0.90);
}

#[test]
fn test_csv_comma() {
    csv_test("csv-comma.csv");
}

#[test]
fn test_csv_semicolon() {
    csv_test("csv-semicolon.csv");
}

#[test]
fn test_csv_tab() {
    csv_test("csv-tab.csv");
}

#[test]
fn test_csv_pipe() {
    csv_test("csv-pipe.csv");
}

#[test]
fn test_csv_single_column() {
    csv_test("csv-single-column.csv");
}

#[test]
fn test_csv_comma_in_cell() {
    csv_test("csv-comma-in-cell.csv");
}

#[test]
fn test_csv_inconsistent_header() {
    csv_test("csv-inconsistent-header.csv");
}

#[test]
fn test_csv_too_few_columns() {
    csv_test("csv-too-few-columns.csv");
}

#[test]
fn test_csv_too_many_columns() {
    csv_test("csv-too-many-columns.csv");
}

#[test]
fn test_csv_empty_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".csv").unwrap();
    std::fs::write(tmp.path(), b"").unwrap();
    let result = run_convert(tmp.path(), &["json"]);
    assert_ne!(result.exit_code, 0, "should fail on empty CSV");
}

#[test]
fn test_csv_invalid_binary() {
    let tmp = tempfile::NamedTempFile::with_suffix(".csv").unwrap();
    std::fs::write(tmp.path(), &[0xFF, 0xFE, 0x00, 0x01, 0x80, 0x81]).unwrap();
    let result = run_convert(tmp.path(), &["json"]);
    assert_ne!(result.exit_code, 0, "should fail on binary content as CSV");
}

#[test]
fn test_csv_exit_code_zero_on_success() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0);
    assert!(result.stderr.is_empty() || !result.stderr.contains("ERROR"));
}
