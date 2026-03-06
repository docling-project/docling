use docling_e2e::helpers::*;

#[test]
fn test_xbrl_mlac() {
    let input = test_data_dir().join("xbrl").join("mlac-20251231.xml");
    if !input.exists() {
        eprintln!("Skipping test_xbrl_mlac: test data not found");
        return;
    }
    let result = run_convert_with_format(&input, &["json", "md"], "xml_xbrl");
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "mlac-20251231", "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
    assert_eq!(
        parsed.get("schema_name").and_then(|v| v.as_str()),
        Some("DoclingDocument")
    );

    let gt_json = "mlac-20251231.xml.json";
    if groundtruth_dir().join(gt_json).exists() {
        let expected_json = read_groundtruth(gt_json);
        assert_json_structural_match(&actual_json, &expected_json);
    } else {
        eprintln!("WARNING: groundtruth file {} not found, skipping comparison", gt_json);
    }

    let actual_md = read_output(&result, "mlac-20251231", "md");
    assert!(!actual_md.is_empty(), "XBRL should produce markdown output");

    let gt_md = "mlac-20251231.xml.md";
    if groundtruth_dir().join(gt_md).exists() {
        let expected_md = read_groundtruth(gt_md);
        assert_md_similar(&actual_md, &expected_md, 0.15);
    } else {
        eprintln!("WARNING: groundtruth file {} not found, skipping comparison", gt_md);
    }
}

#[test]
fn test_xbrl_grve() {
    let input = test_data_dir().join("xbrl").join("grve_10q_htm.xml");
    if !input.exists() {
        eprintln!("Skipping test_xbrl_grve: test data not found");
        return;
    }
    let result = run_convert_with_format(&input, &["json", "md"], "xml_xbrl");
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "grve_10q_htm", "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
    assert_eq!(
        parsed.get("schema_name").and_then(|v| v.as_str()),
        Some("DoclingDocument")
    );

    let gt_json = "grve_10q_htm.xml.json";
    if groundtruth_dir().join(gt_json).exists() {
        let expected_json = read_groundtruth(gt_json);
        assert_json_structural_match(&actual_json, &expected_json);
    } else {
        eprintln!("WARNING: groundtruth file {} not found, skipping comparison", gt_json);
    }

    let actual_md = read_output(&result, "grve_10q_htm", "md");
    assert!(!actual_md.is_empty(), "XBRL should produce markdown output");

    let gt_md = "grve_10q_htm.xml.md";
    if groundtruth_dir().join(gt_md).exists() {
        let expected_md = read_groundtruth(gt_md);
        assert_md_similar(&actual_md, &expected_md, 0.15);
    } else {
        eprintln!("WARNING: groundtruth file {} not found, skipping comparison", gt_md);
    }
}

#[test]
fn test_xbrl_doctags_output() {
    let input = test_data_dir().join("xbrl").join("mlac-20251231.xml");
    if !input.exists() {
        eprintln!("Skipping test_xbrl_doctags_output: test data not found");
        return;
    }
    let result = run_convert_with_format(&input, &["doctags"], "xml_xbrl");
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual = read_output(&result, "mlac-20251231", "doctags");
    assert!(actual.starts_with("<doctags>"));
    assert!(actual.contains("</doctags>"));
}

#[test]
fn test_xbrl_non_xbrl_xml_rejected() {
    let tmp = tempfile::TempDir::new().unwrap();
    let fake_xml = tmp.path().join("not_xbrl.xml");
    std::fs::write(&fake_xml, "<root><child>hello</child></root>").unwrap();

    let out_dir = tmp.path().join("output");
    let output = std::process::Command::new(docling_bin())
        .args([
            "convert",
            fake_xml.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
            "--from",
            "xml_xbrl",
            "--to",
            "json",
            "--abort-on-error",
        ])
        .output()
        .expect("failed to run");

    assert!(
        !output.status.success(),
        "non-XBRL XML should be rejected when using xml_xbrl format"
    );
}
