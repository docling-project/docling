use docling_e2e::helpers::*;

#[test]
fn test_image_webp() {
    let input = test_data_dir().join("webp").join("webp-test.webp");
    let result = run_convert(&input, &["json", "md"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "webp-test", "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
    assert_eq!(
        parsed.get("schema_name").and_then(|v| v.as_str()),
        Some("DoclingDocument")
    );
    assert_eq!(
        parsed.get("name").and_then(|v| v.as_str()),
        Some("webp-test")
    );

    let pictures = parsed.get("pictures").and_then(|v| v.as_array());
    assert!(
        pictures.map(|p| !p.is_empty()).unwrap_or(false),
        "Image backend should produce at least one picture item"
    );

    let pic = &pictures.unwrap()[0];
    let image = pic.get("image");
    assert!(image.is_some(), "Picture should have image metadata");
    let image = image.unwrap();
    assert!(
        image.get("size").is_some(),
        "Image metadata should include size"
    );
    let size = image.get("size").unwrap();
    assert!(
        size.get("width")
            .and_then(|v| v.as_f64())
            .map(|w| w > 0.0)
            .unwrap_or(false),
        "Image width should be > 0"
    );
    assert!(
        size.get("height")
            .and_then(|v| v.as_f64())
            .map(|h| h > 0.0)
            .unwrap_or(false),
        "Image height should be > 0"
    );

    let actual_md = read_output(&result, "webp-test", "md");
    assert!(!actual_md.is_empty(), "Image backend should produce output");
}

#[test]
fn test_image_png() {
    let input = test_data_dir().join("2305.03393v1-pg9-img.png");
    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "2305.03393v1-pg9-img", "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
    assert_eq!(
        parsed.get("schema_name").and_then(|v| v.as_str()),
        Some("DoclingDocument")
    );

    let pictures = parsed.get("pictures").and_then(|v| v.as_array());
    assert!(
        pictures.map(|p| !p.is_empty()).unwrap_or(false),
        "Image backend should produce at least one picture item"
    );

    let pic = &pictures.unwrap()[0];
    assert!(
        pic.get("image").is_some(),
        "Picture should have image metadata for PNG"
    );
}

#[test]
fn test_image_tiff() {
    let input = test_data_dir().join("tiff").join("2206.01062.tif");
    let result = run_convert(&input, &["json"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "2206.01062", "json");
    let parsed: serde_json::Value =
        serde_json::from_str(&actual_json).expect("output should be valid JSON");
    assert_eq!(
        parsed.get("schema_name").and_then(|v| v.as_str()),
        Some("DoclingDocument")
    );
}

#[test]
fn test_image_corrupt_file() {
    let tmp = tempfile::TempDir::new().unwrap();
    let bad_img = tmp.path().join("corrupt.png");
    std::fs::write(&bad_img, b"this is not a valid PNG file").unwrap();

    let result = run_convert(&bad_img, &["json"]);
    assert_eq!(
        result.exit_code, 0,
        "corrupt image should still produce a document (with no image metadata)"
    );

    let actual_json = read_output(&result, "corrupt", "json");
    let parsed: serde_json::Value = serde_json::from_str(&actual_json).unwrap();
    let pictures = parsed.get("pictures").and_then(|v| v.as_array()).unwrap();
    assert!(!pictures.is_empty(), "should still have a picture item");
    assert!(
        pictures[0].get("image").is_none() || pictures[0].get("image").unwrap().is_null(),
        "corrupt image should have null/missing image metadata"
    );
}

#[test]
fn test_image_doctags_output() {
    let input = test_data_dir().join("2305.03393v1-pg9-img.png");
    let result = run_convert(&input, &["doctags"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual = read_output(&result, "2305.03393v1-pg9-img", "doctags");
    assert!(actual.contains("<picture/>"), "should contain picture tag");
}

#[test]
fn test_image_vtt_output() {
    let input = test_data_dir().join("2305.03393v1-pg9-img.png");
    let result = run_convert(&input, &["vtt"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual = read_output(&result, "2305.03393v1-pg9-img", "vtt");
    assert!(actual.starts_with("WEBVTT"));
}
