use docling_e2e::helpers::*;

fn md_test(filename: &str) {
    let input = test_data_dir().join("md").join(filename);
    let stem = filename.strip_suffix(".md").unwrap_or(filename);

    let result = run_convert(&input, &["md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_md = read_output(&result, stem, "md");
    let gt_md_name = format!("{}.md", filename);
    let expected_md = read_groundtruth(&gt_md_name);
    assert_md_similar(&actual_md, &expected_md, 0.42);
}

#[test]
fn test_md_wiki() {
    md_test("wiki.md");
}

#[test]
fn test_md_blocks() {
    md_test("blocks.md");
}

#[test]
fn test_md_duck() {
    md_test("duck.md");
}

#[test]
fn test_md_mixed() {
    md_test("mixed.md");
}

#[test]
fn test_md_nested() {
    md_test("nested.md");
}

#[test]
fn test_md_inline_and_formatting() {
    md_test("inline_and_formatting.md");
}

#[test]
fn test_md_escaped_characters() {
    md_test("escaped_characters.md");
}

#[test]
fn test_md_ending_with_table() {
    md_test("ending_with_table.md");
}

#[test]
fn test_md_mixed_without_h1() {
    md_test("mixed_without_h1.md");
}
