use docling_e2e::helpers::*;

fn asciidoc_test(filename: &str) {
    let input = test_data_dir().join("asciidoc").join(filename);
    let stem = filename.strip_suffix(".asciidoc").unwrap_or(filename);

    let result = run_convert(&input, &["md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_md = read_output(&result, stem, "md");
    let gt_md_name = format!("{}.md", filename);
    if std::path::Path::new(&groundtruth_dir().join(&gt_md_name)).exists() {
        let expected_md = read_groundtruth(&gt_md_name);
        assert_md_similar(&actual_md, &expected_md, 0.45);
    } else {
        assert!(!actual_md.is_empty(), "Output should not be empty");
    }
}

#[test]
fn test_asciidoc_01() {
    asciidoc_test("test_01.asciidoc");
}

#[test]
fn test_asciidoc_02() {
    asciidoc_test("test_02.asciidoc");
}

#[test]
fn test_asciidoc_03() {
    asciidoc_test("test_03.asciidoc");
}
