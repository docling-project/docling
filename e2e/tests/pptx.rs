use docling_e2e::helpers::*;

fn pptx_test(filename: &str) {
    pptx_test_with_threshold(filename, 0.99);
}

fn pptx_test_with_threshold(filename: &str, md_threshold: f64) {
    let input = test_data_dir().join("pptx").join(filename);
    let stem = filename.strip_suffix(".pptx").unwrap_or(filename);

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
        assert!(!actual_md.is_empty(), "PPTX should produce markdown output");
    }
}

#[test]
fn test_pptx_sample() {
    pptx_test_with_threshold("powerpoint_sample.pptx", 0.99);
}

#[test]
fn test_pptx_bad_text() {
    pptx_test("powerpoint_bad_text.pptx");
}

#[test]
fn test_pptx_with_image() {
    pptx_test_with_threshold("powerpoint_with_image.pptx", 0.99);
}

#[test]
fn test_pptx_issue_2663() {
    pptx_test("powerpoint_issue_2663.pptx");
}

#[test]
fn test_pptx_invalid_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".pptx").unwrap();
    std::fs::write(tmp.path(), b"this is not a pptx file").unwrap();
    let result = run_convert_expect_failure(tmp.path(), &["json"]);
    assert_ne!(result.exit_code, 0, "should fail on invalid PPTX");
}

#[test]
fn test_pptx_empty_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".pptx").unwrap();
    std::fs::write(tmp.path(), b"").unwrap();
    let result = run_convert_expect_failure(tmp.path(), &["json"]);
    assert_ne!(result.exit_code, 0, "should fail on empty PPTX");
}
