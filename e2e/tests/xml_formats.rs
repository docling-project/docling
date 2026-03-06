use docling_e2e::helpers::*;

// ---- JATS tests ----

fn assert_jats_fixture(fixture: &str, stem: &str) {
    let input = test_data_dir().join("jats").join(fixture);
    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(
        result.exit_code, 0,
        "convert failed for {}: {}",
        fixture, result.stderr
    );

    let gt_json_name = format!("{}.json", fixture);
    if groundtruth_dir().join(&gt_json_name).exists() {
        let actual_json = read_output(&result, stem, "json");
        let expected_json = read_groundtruth(&gt_json_name);
        assert_json_lenient_structural_match(&actual_json, &expected_json);
    }

    let gt_md_name = format!("{}.md", fixture);
    let actual_md = read_output(&result, stem, "md");
    assert!(
        !actual_md.is_empty(),
        "JATS {} should produce MD output",
        fixture
    );
    if groundtruth_dir().join(&gt_md_name).exists() {
        let expected_md = read_groundtruth(&gt_md_name);
        assert_md_similar(&actual_md, &expected_md, 0.20);
    }
}

#[test]
fn test_jats_elife() {
    assert_jats_fixture("elife-56337.nxml", "elife-56337");
}

#[test]
fn test_jats_pone() {
    assert_jats_fixture("pone.0234687.nxml", "pone.0234687");
}

#[test]
fn test_jats_pntd() {
    assert_jats_fixture("pntd.0008301.nxml", "pntd.0008301");
}

// ---- USPTO tests ----

fn assert_uspto_fixture(fixture: &str, stem: &str) {
    let input = test_data_dir().join("uspto").join(fixture);
    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(
        result.exit_code, 0,
        "convert failed for {}: {}",
        fixture, result.stderr
    );

    let gt_json_name = format!("{}.json", stem);
    if groundtruth_dir().join(&gt_json_name).exists() {
        let actual_json = read_output(&result, stem, "json");
        let expected_json = read_groundtruth(&gt_json_name);
        assert_json_lenient_structural_match(&actual_json, &expected_json);
    }

    let gt_md_name = format!("{}.md", stem);
    let actual_md = read_output(&result, stem, "md");
    assert!(
        !actual_md.is_empty(),
        "USPTO {} should produce MD output",
        fixture
    );
    if groundtruth_dir().join(&gt_md_name).exists() {
        let expected_md = read_groundtruth(&gt_md_name);
        assert_md_similar(&actual_md, &expected_md, 0.15);
    }
}

#[test]
fn test_uspto_ipa_2018() {
    assert_uspto_fixture("ipa20180000016.xml", "ipa20180000016");
}

#[test]
fn test_uspto_ipa_2020() {
    assert_uspto_fixture("ipa20200022300.xml", "ipa20200022300");
}

#[test]
fn test_uspto_pa_2001() {
    assert_uspto_fixture("pa20010031492.xml", "pa20010031492");
}

#[test]
fn test_uspto_pg_grant() {
    assert_uspto_fixture("pg06442728.xml", "pg06442728");
}

#[test]
fn test_uspto_aps() {
    let input = test_data_dir().join("uspto").join("pftaps057006474.txt");
    let result = run_convert_with_format(&input, &["json", "md"], "xml_uspto");
    assert_eq!(
        result.exit_code, 0,
        "convert failed for APS: {}",
        result.stderr
    );

    let actual_md = read_output(&result, "pftaps057006474", "md");
    assert!(!actual_md.is_empty(), "APS patent should produce MD output");

    let gt_json_name = "pftaps057006474.json";
    if groundtruth_dir().join(gt_json_name).exists() {
        let actual_json = read_output(&result, "pftaps057006474", "json");
        let expected_json = read_groundtruth(gt_json_name);
        assert_json_lenient_structural_match(&actual_json, &expected_json);
    }

    let gt_md_name = "pftaps057006474.md";
    if groundtruth_dir().join(gt_md_name).exists() {
        let expected_md = read_groundtruth(gt_md_name);
        assert_md_similar(&actual_md, &expected_md, 0.15);
    }
}

// ---- Docling JSON tests ----

#[test]
fn test_json_docling_roundtrip() {
    let csv_input = test_data_dir().join("csv").join("csv-comma.csv");
    let first = run_convert(&csv_input, &["json"]);
    assert_eq!(first.exit_code, 0);

    let first_json = read_output(&first, "csv-comma", "json");

    let json_path = first.output_dir.path().join("csv-comma.json");
    let second = run_convert(&json_path, &["json"]);
    assert_eq!(second.exit_code, 0);

    let roundtrip_json = read_output(&second, "csv-comma", "json");

    let first_val: serde_json::Value =
        serde_json::from_str(&first_json).expect("first JSON should be valid");
    let second_val: serde_json::Value =
        serde_json::from_str(&roundtrip_json).expect("roundtrip JSON should be valid");

    assert_eq!(
        first_val.get("schema_name"),
        second_val.get("schema_name"),
        "schema_name changed after roundtrip"
    );
    assert_eq!(
        first_val.get("name"),
        second_val.get("name"),
        "name changed after roundtrip"
    );

    let first_texts = first_val.get("texts").and_then(|v| v.as_array());
    let second_texts = second_val.get("texts").and_then(|v| v.as_array());
    if let (Some(ft), Some(st)) = (first_texts, second_texts) {
        assert_eq!(ft.len(), st.len(), "texts count changed after roundtrip");
    }

    let first_tables = first_val.get("tables").and_then(|v| v.as_array());
    let second_tables = second_val.get("tables").and_then(|v| v.as_array());
    if let (Some(ft), Some(st)) = (first_tables, second_tables) {
        assert_eq!(ft.len(), st.len(), "tables count changed after roundtrip");
    }
}

#[test]
fn test_json_docling_invalid_input() {
    let dir = tempfile::tempdir().expect("tempdir");
    let bad_json = dir.path().join("bad.json");
    std::fs::write(&bad_json, "{ invalid json }").expect("write");
    let result = run_convert_expect_failure(&bad_json, &["json"]);
    assert_ne!(
        result.exit_code, 0,
        "invalid JSON should fail: {}",
        result.stderr
    );
}
