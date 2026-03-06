use docling_e2e::helpers::*;

fn xlsx_test(filename: &str) {
    xlsx_test_with_threshold(filename, 0.95);
}

fn xlsx_test_with_threshold(filename: &str, md_threshold: f64) {
    let input = test_data_dir().join("xlsx").join(filename);
    let stem = filename
        .strip_suffix(".xlsx")
        .or_else(|| filename.strip_suffix(".xlsm"))
        .unwrap_or(filename);

    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, stem, "json");
    assert_valid_docling_document(&actual_json);

    let gt_json_name = format!("{}.json", filename);
    if groundtruth_dir().join(&gt_json_name).exists() {
        let expected_json = read_groundtruth(&gt_json_name);
        assert_json_strict_structural_match(&actual_json, &expected_json);
    }

    let actual_md = read_output(&result, stem, "md");
    let gt_md_name = format!("{}.md", filename);
    if groundtruth_dir().join(&gt_md_name).exists() {
        let expected_md = read_groundtruth(&gt_md_name);
        assert_md_similar(&actual_md, &expected_md, md_threshold);
    } else {
        assert!(!actual_md.is_empty(), "XLSX should produce markdown output");
    }
}

#[test]
fn test_xlsx_01() {
    xlsx_test("xlsx_01.xlsx");
}

#[test]
fn test_xlsx_02_sample_sales() {
    xlsx_test("xlsx_02_sample_sales_data.xlsm");
}

#[test]
fn test_xlsx_03_chartsheet() {
    xlsx_test_with_threshold("xlsx_03_chartsheet.xlsx", 0.99);
}

#[test]
fn test_xlsx_04_inflated() {
    xlsx_test("xlsx_04_inflated.xlsx");
}

#[test]
fn test_xlsx_05_table_with_title() {
    xlsx_test("xlsx_05_table_with_title.xlsx");
}

#[test]
fn test_xlsx_06_edge_cases() {
    xlsx_test("xlsx_06_edge_cases_.xlsx");
}

#[test]
fn test_xlsx_07_gap_tolerance() {
    xlsx_test("xlsx_07_gap_tolerance_.xlsx");
}

#[test]
fn test_xlsx_08_one_cell_anchor() {
    xlsx_test_with_threshold("xlsx_08_one_cell_anchor.xlsx", 0.98);
}

#[test]
fn test_xlsx_invalid_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".xlsx").unwrap();
    std::fs::write(tmp.path(), b"this is not an xlsx file").unwrap();
    let result = run_convert_expect_failure(tmp.path(), &["json"]);
    assert_ne!(result.exit_code, 0, "should fail on invalid XLSX");
}

#[test]
fn test_xlsx_empty_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".xlsx").unwrap();
    std::fs::write(tmp.path(), b"").unwrap();
    let result = run_convert_expect_failure(tmp.path(), &["json"]);
    assert_ne!(result.exit_code, 0, "should fail on empty XLSX");
}
