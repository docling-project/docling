use docling_e2e::helpers::*;

fn docx_test(filename: &str) {
    docx_test_with_threshold(filename, 0.90);
}

fn docx_test_with_threshold(filename: &str, md_threshold: f64) {
    let input = test_data_dir().join("docx").join(filename);
    let stem = filename.strip_suffix(".docx").unwrap_or(filename);

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
    }
}

// --- Core feature tests (high fidelity expected) ---

#[test]
fn test_docx_lorem_ipsum() {
    docx_test_with_threshold("lorem_ipsum.docx", 0.95);
}

#[test]
fn test_docx_word_sample() {
    docx_test_with_threshold("word_sample.docx", 0.95);
}

#[test]
fn test_docx_word_tables() {
    docx_test_with_threshold("word_tables.docx", 0.98);
}

#[test]
fn test_docx_unit_test_headers() {
    docx_test("unit_test_headers.docx");
}

#[test]
fn test_docx_unit_test_lists() {
    docx_test_with_threshold("unit_test_lists.docx", 0.95);
}

#[test]
fn test_docx_headers_numbered() {
    docx_test("unit_test_headers_numbered.docx");
}

#[test]
fn test_docx_list_after_num_headers() {
    docx_test_with_threshold("list_after_num_headers.docx", 0.95);
}

#[test]
fn test_docx_word_comments() {
    docx_test_with_threshold("word_comments.docx", 0.95);
}

// --- Formatting, textboxes, rich cells ---

#[test]
fn test_docx_unit_test_formatting() {
    docx_test_with_threshold("unit_test_formatting.docx", 0.90);
}

#[test]
fn test_docx_tablecell() {
    docx_test_with_threshold("tablecell.docx", 0.96);
}

#[test]
fn test_docx_textbox() {
    docx_test_with_threshold("textbox.docx", 0.75);
}

#[test]
fn test_docx_rich_cells() {
    docx_test_with_threshold("docx_rich_cells.docx", 0.60);
}

#[test]
fn test_docx_equations() {
    docx_test("equations.docx");
}

#[test]
fn test_docx_table_with_equations() {
    docx_test("table_with_equations.docx");
}

// --- Image-heavy fixtures ---

#[test]
fn test_docx_word_image_anchors() {
    docx_test_with_threshold("word_image_anchors.docx", 0.95);
}

#[test]
fn test_docx_test_emf() {
    docx_test_with_threshold("test_emf_docx.docx", 0.95);
}

#[test]
fn test_docx_drawingml() {
    docx_test_with_threshold("drawingml.docx", 0.45);
}

#[test]
fn test_docx_grouped_images() {
    docx_test("docx_grouped_images.docx");
}

// --- Error handling tests ---

#[test]
fn test_docx_invalid_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".docx").unwrap();
    std::fs::write(tmp.path(), b"this is not a docx file").unwrap();
    let result = run_convert_expect_failure(tmp.path(), &["json"]);
    assert_ne!(result.exit_code, 0, "should fail on invalid DOCX");
}

#[test]
fn test_docx_empty_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".docx").unwrap();
    std::fs::write(tmp.path(), b"").unwrap();
    let result = run_convert_expect_failure(tmp.path(), &["json"]);
    assert_ne!(result.exit_code, 0, "should fail on empty DOCX");
}
