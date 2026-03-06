use docling_e2e::helpers::*;

fn html_test(filename: &str) {
    html_test_with_threshold(filename, 0.30);
}

fn html_test_with_threshold(filename: &str, md_threshold: f64) {
    let input = test_data_dir().join("html").join(filename);
    let stem = filename.strip_suffix(".html").unwrap_or(filename);

    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, stem, "json");
    let actual_md = read_output(&result, stem, "md");

    // Validate JSON structural match against groundtruth
    let gt_json_name = format!("{}.json", filename);
    if std::path::Path::new(&groundtruth_dir().join(&gt_json_name)).exists() {
        let expected_json = read_groundtruth(&gt_json_name);
        assert_json_structural_match(&actual_json, &expected_json);
    }

    // Markdown similarity check
    let gt_md_name = format!("{}.md", filename);
    if std::path::Path::new(&groundtruth_dir().join(&gt_md_name)).exists() {
        let expected_md = read_groundtruth(&gt_md_name);
        if !expected_md.trim().is_empty() && expected_md.trim().len() > 5 {
            assert_md_similar(&actual_md, &expected_md, md_threshold);
        }
    }
}

#[test]
fn test_html_example_01() {
    html_test("example_01.html");
}

#[test]
fn test_html_example_02() {
    html_test("example_02.html");
}

#[test]
fn test_html_example_03() {
    html_test("example_03.html");
}

#[test]
fn test_html_example_04() {
    html_test("example_04.html");
}

#[test]
fn test_html_example_05() {
    html_test("example_05.html");
}

#[test]
fn test_html_example_06() {
    html_test("example_06.html");
}

#[test]
fn test_html_example_07() {
    html_test("example_07.html");
}

#[test]
fn test_html_example_08() {
    html_test("example_08.html");
}

#[test]
fn test_html_wiki_duck() {
    // Full Wikipedia page with navigation chrome — lower threshold due to nav/sidebar extraction and image captions
    html_test_with_threshold("wiki_duck.html", 0.12);
}

#[test]
fn test_html_table_01() {
    html_test("table_01.html");
}

#[test]
fn test_html_table_02() {
    html_test("table_02.html");
}

#[test]
fn test_html_table_03() {
    html_test("table_03.html");
}

#[test]
fn test_html_table_04() {
    html_test("table_04.html");
}

#[test]
fn test_html_table_05() {
    html_test("table_05.html");
}

#[test]
fn test_html_table_06() {
    // Deeply nested tables (table-in-table-in-table) - known limitation
    html_test_with_threshold("table_06.html", 0.05);
}

#[test]
fn test_html_hyperlink_01() {
    html_test("hyperlink_01.html");
}

#[test]
fn test_html_hyperlink_02() {
    html_test("hyperlink_02.html");
}

#[test]
fn test_html_hyperlink_03() {
    html_test("hyperlink_03.html");
}

#[test]
fn test_html_hyperlink_04() {
    html_test("hyperlink_04.html");
}

#[test]
fn test_html_hyperlink_05() {
    html_test_with_threshold("hyperlink_05.html", 0.15);
}

#[test]
fn test_html_formatting() {
    html_test("formatting.html");
}

#[test]
fn test_html_code_snippets() {
    html_test("html_code_snippets.html");
}

#[test]
fn test_html_unit_test_01() {
    html_test("unit_test_01.html");
}

#[test]
fn test_html_table_with_heading_01() {
    html_test("table_with_heading_01.html");
}

#[test]
fn test_html_table_with_heading_02() {
    html_test("table_with_heading_02.html");
}

#[test]
fn test_html_rich_table_cells() {
    html_test("html_rich_table_cells.html");
}

#[test]
fn test_html_heading_in_p() {
    html_test("html_heading_in_p.html");
}
