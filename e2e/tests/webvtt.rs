use docling_e2e::helpers::*;

fn vtt_test(filename: &str) {
    let input = test_data_dir().join("webvtt").join(filename);
    let stem = filename.strip_suffix(".vtt").unwrap_or(filename);

    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, stem, "json");
    let gt_json_name = format!("{}.json", filename);
    if std::path::Path::new(&groundtruth_dir().join(&gt_json_name)).exists() {
        let expected_json = read_groundtruth(&gt_json_name);
        assert_json_structural_match(&actual_json, &expected_json);
    }

    let actual_md = read_output(&result, stem, "md");
    let gt_md_name = format!("{}.md", filename);
    if std::path::Path::new(&groundtruth_dir().join(&gt_md_name)).exists() {
        let expected_md = read_groundtruth(&gt_md_name);
        assert_md_similar(&actual_md, &expected_md, 0.60);
    } else {
        assert!(!actual_md.is_empty(), "Output should not be empty");
    }
}

#[test]
fn test_webvtt_01() {
    vtt_test("webvtt_example_01.vtt");
}

#[test]
fn test_webvtt_02() {
    vtt_test("webvtt_example_02.vtt");
}

#[test]
fn test_webvtt_03() {
    vtt_test("webvtt_example_03.vtt");
}

#[test]
fn test_webvtt_04() {
    vtt_test("webvtt_example_04.vtt");
}
