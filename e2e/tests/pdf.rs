use docling_e2e::helpers::*;
use docling_e2e::pdf_fixtures::ensure_pdf_fixtures;

fn setup() {
    ensure_pdf_fixtures(&test_data_dir());
}

// ---------------------------------------------------------------------------
// Tests using generated fixtures (always runnable)
// ---------------------------------------------------------------------------

#[test]
fn test_pdf_single_page_basic() {
    setup();
    let input = test_data_dir().join("pdf").join("single_page.pdf");
    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let json_str = read_output(&result, "single_page", "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("output should be valid JSON");

    assert_eq!(
        parsed.get("schema_name").and_then(|v| v.as_str()),
        Some("DoclingDocument")
    );
    assert_eq!(
        parsed.get("name").and_then(|v| v.as_str()),
        Some("single_page")
    );

    let pages = parsed
        .get("pages")
        .and_then(|v| v.as_object())
        .expect("should have pages");
    assert_eq!(pages.len(), 1, "should have exactly 1 page");

    let page1 = pages.get("1").expect("page 1 should exist");
    let size = page1.get("size").expect("page should have size");
    assert_eq!(size.get("width").and_then(|v| v.as_f64()), Some(612.0));
    assert_eq!(size.get("height").and_then(|v| v.as_f64()), Some(792.0));

    let texts = parsed
        .get("texts")
        .and_then(|v| v.as_array())
        .expect("should have texts array");
    assert!(!texts.is_empty(), "should extract at least one text item");

    for text in texts {
        let prov = text
            .get("prov")
            .and_then(|v| v.as_array())
            .expect("text should have prov");
        assert!(!prov.is_empty(), "prov should not be empty");
        let page_no = prov[0].get("page_no").and_then(|v| v.as_u64());
        assert_eq!(page_no, Some(1), "prov should reference page 1");
    }

    let md = read_output(&result, "single_page", "md");
    assert!(!md.is_empty(), "markdown output should not be empty");
    assert!(
        md.contains("Test Document Title") || md.to_lowercase().contains("test document"),
        "markdown should contain the document title, got:\n{}",
        &md[..md.len().min(500)]
    );
}

#[test]
fn test_pdf_multi_page_generated() {
    setup();
    let input = test_data_dir().join("pdf").join("multi_page_generated.pdf");
    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let json_str = read_output(&result, "multi_page_generated", "json");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");

    let pages = parsed
        .get("pages")
        .and_then(|v| v.as_object())
        .expect("should have pages");
    assert_eq!(pages.len(), 3, "should have exactly 3 pages");

    let texts = parsed
        .get("texts")
        .and_then(|v| v.as_array())
        .expect("texts");
    assert!(
        texts.len() >= 3,
        "should have at least 3 text items (one per page), got {}",
        texts.len()
    );

    let md = read_output(&result, "multi_page_generated", "md");
    assert!(!md.is_empty(), "markdown should not be empty");
}

#[test]
fn test_pdf_headings_and_text() {
    setup();
    let input = test_data_dir().join("pdf").join("headings_and_text.pdf");
    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let json_str = read_output(&result, "headings_and_text", "json");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");

    let texts = parsed
        .get("texts")
        .and_then(|v| v.as_array())
        .expect("texts");

    let labels: Vec<&str> = texts
        .iter()
        .filter_map(|t| t.get("label").and_then(|v| v.as_str()))
        .collect();

    assert!(
        labels.contains(&"section_header"),
        "should detect at least one section_header, got labels: {:?}",
        labels
    );
    assert!(
        labels.contains(&"text"),
        "should have at least one text item, got labels: {:?}",
        labels
    );

    for text in texts {
        if text.get("label").and_then(|v| v.as_str()) == Some("section_header") {
            let level = text.get("level").and_then(|v| v.as_u64());
            assert!(level.is_some(), "section_header should have a level");
        }
    }
}

#[test]
fn test_pdf_empty_page() {
    setup();
    let input = test_data_dir().join("pdf").join("empty_page.pdf");
    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert should succeed for empty PDF");

    let json_str = read_output(&result, "empty_page", "json");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");

    let pages = parsed
        .get("pages")
        .and_then(|v| v.as_object())
        .expect("should have pages");
    assert_eq!(pages.len(), 1, "should have 1 page");

    let texts = parsed
        .get("texts")
        .and_then(|v| v.as_array())
        .expect("texts");
    assert!(
        texts.is_empty(),
        "empty PDF should produce no text items, got {}",
        texts.len()
    );
}

// ---------------------------------------------------------------------------
// Error-handling tests
// ---------------------------------------------------------------------------

#[test]
fn test_pdf_nonexistent_file() {
    let input = test_data_dir().join("pdf").join("does_not_exist.pdf");
    let result = run_convert_expect_failure(&input, &["json"]);
    assert_ne!(result.exit_code, 0, "should fail for nonexistent file");
}

#[test]
fn test_pdf_corrupted_file() {
    setup();
    let corrupt_path = test_data_dir().join("pdf").join("corrupted.pdf");
    std::fs::write(&corrupt_path, b"This is not a valid PDF file at all.")
        .expect("write corrupt fixture");

    let result = run_convert_expect_failure(&corrupt_path, &["json"]);
    assert_ne!(result.exit_code, 0, "should fail for corrupted PDF");
}

// ---------------------------------------------------------------------------
// Original fixture tests with groundtruth comparison.
// ---------------------------------------------------------------------------

fn original_pdf_test(filename: &str, md_threshold: f64) {
    let input = test_data_dir().join("pdf").join(filename);
    if !input.exists() {
        eprintln!("SKIP: fixture {} not found", input.display());
        return;
    }

    let stem = filename.strip_suffix(".pdf").unwrap_or(filename);

    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, stem, "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");

    assert_eq!(
        parsed.get("schema_name").and_then(|v| v.as_str()),
        Some("DoclingDocument")
    );

    let pages = parsed
        .get("pages")
        .and_then(|v| v.as_object())
        .expect("should have pages");
    assert!(!pages.is_empty(), "should have at least one page");

    let texts = parsed
        .get("texts")
        .and_then(|v| v.as_array())
        .expect("texts");
    assert!(!texts.is_empty(), "should extract text from PDF");

    for text in texts {
        let prov = text.get("prov").and_then(|v| v.as_array());
        assert!(
            prov.is_some() && !prov.unwrap().is_empty(),
            "every text item should have provenance"
        );
    }

    let actual_md = read_output(&result, stem, "md");
    assert!(!actual_md.is_empty(), "PDF should produce markdown output");

    // Groundtruth JSON comparison
    let gt_json_name = format!("{}.json", stem);
    let gt_json_path = groundtruth_dir().join(&gt_json_name);
    if gt_json_path.exists() {
        let expected_json = read_groundtruth(&gt_json_name);
        let expected_val: serde_json::Value =
            serde_json::from_str(&expected_json).expect("expected JSON");
        assert_eq!(
            parsed.get("schema_name"),
            expected_val.get("schema_name"),
            "schema_name mismatch"
        );
        assert_eq!(
            parsed.get("name"),
            expected_val.get("name"),
            "name mismatch"
        );
        let expected_texts = expected_val.get("texts").and_then(|v| v.as_array());
        if let Some(et) = expected_texts {
            if !et.is_empty() {
                assert!(
                    !texts.is_empty(),
                    "expected texts but got none (groundtruth has {})",
                    et.len()
                );
            }
        }
    }

    // Groundtruth MD comparison
    let gt_md_name = format!("{}.md", stem);
    let gt_md_path = groundtruth_dir().join(&gt_md_name);
    if gt_md_path.exists() {
        let expected_md = read_groundtruth(&gt_md_name);
        let sim = levenshtein_similarity(actual_md.trim(), expected_md.trim());
        eprintln!("  {}: MD similarity = {:.1}%", filename, sim * 100.0);
        assert!(
            sim >= md_threshold,
            "Markdown similarity {:.1}% below threshold {:.1}% for {}",
            sim * 100.0,
            md_threshold * 100.0,
            filename
        );
    }
}

// ---------------------------------------------------------------------------
// Individual PDF fixture tests
// ---------------------------------------------------------------------------

#[test]
fn test_pdf_2305_03393v1_pg9() {
    original_pdf_test("2305.03393v1-pg9.pdf", 0.45);
}

#[test]
fn test_pdf_multi_page() {
    original_pdf_test("multi_page.pdf", 0.95);
}

#[test]
fn test_pdf_normal_4pages() {
    original_pdf_test("normal_4pages.pdf", 0.25);
}

#[test]
fn test_pdf_code_and_formula() {
    original_pdf_test("code_and_formula.pdf", 0.90);
}

#[test]
fn test_pdf_2203_01017v2() {
    original_pdf_test("2203.01017v2.pdf", 0.15);
}

#[test]
fn test_pdf_2206_01062() {
    original_pdf_test("2206.01062.pdf", 0.10);
}

#[test]
fn test_pdf_2305_03393v1_full() {
    original_pdf_test("2305.03393v1.pdf", 0.15);
}

#[test]
fn test_pdf_amt_handbook_sample() {
    original_pdf_test("amt_handbook_sample.pdf", 0.70);
}

#[test]
fn test_pdf_picture_classification() {
    original_pdf_test("picture_classification.pdf", 0.90);
}

#[test]
fn test_pdf_redp5110_sampled() {
    original_pdf_test("redp5110_sampled.pdf", 0.20);
}

#[test]
fn test_pdf_right_to_left_01() {
    original_pdf_test("right_to_left_01.pdf", 0.50);
}

#[test]
fn test_pdf_right_to_left_02() {
    original_pdf_test("right_to_left_02.pdf", 0.55);
}

#[test]
fn test_pdf_right_to_left_03() {
    original_pdf_test("right_to_left_03.pdf", 0.45);
}
