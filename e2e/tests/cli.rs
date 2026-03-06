use std::fs;
use std::process::Command;

use docling_e2e::helpers::*;

#[test]
fn test_cli_help() {
    let output = Command::new(docling_bin())
        .arg("--help")
        .output()
        .expect("failed to run");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Document conversion tool"));
}

#[test]
fn test_cli_version() {
    let output = Command::new(docling_bin())
        .arg("--version")
        .output()
        .expect("failed to run");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("docling-rs"));
}

#[test]
fn test_cli_convert_help() {
    let output = Command::new(docling_bin())
        .args(["convert", "--help"])
        .output()
        .expect("failed to run");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--to"));
    assert!(stdout.contains("--output"));
    assert!(stdout.contains("--abort-on-error"));
    assert!(stdout.contains("--document-timeout"));
    assert!(stdout.contains("--num-threads"));
    assert!(stdout.contains("--image-export-mode"));
}

#[test]
fn test_cli_convert_missing_source() {
    let output = Command::new(docling_bin())
        .args(["convert"])
        .output()
        .expect("failed to run");
    assert!(!output.status.success());
}

#[test]
fn test_cli_convert_multiple_formats() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let result = run_convert(&input, &["json", "md", "yaml"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    assert!(result.output_dir.path().join("csv-comma.json").exists());
    assert!(result.output_dir.path().join("csv-comma.md").exists());
    assert!(result.output_dir.path().join("csv-comma.yaml").exists());
}

#[test]
fn test_cli_convert_nonexistent_file() {
    let output = Command::new(docling_bin())
        .args([
            "convert",
            "/nonexistent/file.csv",
            "-o",
            "/tmp/docling-noexist",
        ])
        .output()
        .expect("failed to run");
    assert!(!output.status.success());
}

// --- DocTags output format ---

#[test]
fn test_cli_convert_doctags_output() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let result = run_convert(&input, &["doctags"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual = read_output(&result, "csv-comma", "doctags");
    assert!(
        actual.starts_with("<doctags>"),
        "should start with <doctags> root"
    );
    assert!(actual.contains("</doctags>"), "should end with </doctags>");
    assert!(actual.contains("<table>"), "CSV should produce a table");
}

#[test]
fn test_cli_convert_doctags_escaping() {
    let input = test_data_dir().join("html").join("wiki_duck.html");
    if input.exists() {
        let result = run_convert(&input, &["doctags"]);
        assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

        let actual = read_output(&result, "wiki_duck", "doctags");
        assert!(
            !actual.contains("<<"),
            "doctags should not contain unescaped angle brackets in text"
        );
    }
}

// --- VTT output format ---

#[test]
fn test_cli_convert_vtt_output() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let result = run_convert(&input, &["vtt"]);
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual = read_output(&result, "csv-comma", "vtt");
    assert!(
        actual.starts_with("WEBVTT"),
        "VTT output should start with WEBVTT header"
    );
    assert!(
        actual.contains("-->"),
        "VTT output should contain timestamp arrows"
    );
}

#[test]
fn test_cli_convert_vtt_from_webvtt() {
    let vtt_files: Vec<_> = fs::read_dir(test_data_dir().join("webvtt"))
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "vtt").unwrap_or(false))
        .collect();

    if let Some(entry) = vtt_files.first() {
        let result = run_convert(&entry.path(), &["vtt"]);
        assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

        let stem = entry
            .path()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let actual = read_output(&result, &stem, "vtt");
        assert!(actual.starts_with("WEBVTT"));
    }
}

// --- --from explicit format ---

#[test]
fn test_cli_from_explicit_format() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let result = run_convert_with_format(&input, &["json"], "csv");
    assert_eq!(result.exit_code, 0, "convert failed: {}", result.stderr);

    let actual_json = read_output(&result, "csv-comma", "json");
    let parsed: serde_json::Value = serde_json::from_str(&actual_json).unwrap();
    assert_eq!(
        parsed.get("schema_name").and_then(|v| v.as_str()),
        Some("DoclingDocument")
    );
}

// --- --abort-on-error ---

#[test]
fn test_cli_abort_on_error_stops_on_failure() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Use a corrupt .docx file (not a valid zip) to trigger a real conversion error
    let bad_file = tmp.path().join("bad.docx");
    fs::write(&bad_file, "this is not a valid docx file").unwrap();
    let good_file = test_data_dir().join("csv").join("csv-comma.csv");

    let output = Command::new(docling_bin())
        .args([
            "convert",
            bad_file.to_str().unwrap(),
            good_file.to_str().unwrap(),
            "--abort-on-error",
            "-o",
            tmp.path().to_str().unwrap(),
            "--to",
            "json",
        ])
        .output()
        .expect("failed to run");

    assert!(
        !output.status.success(),
        "should fail with abort-on-error when first file fails"
    );
}

#[test]
fn test_cli_no_abort_continues_on_failure() {
    let tmp = tempfile::TempDir::new().unwrap();
    let bad_file = tmp.path().join("bad.docx");
    fs::write(&bad_file, "this is not a valid docx file").unwrap();
    let good_file = test_data_dir().join("csv").join("csv-comma.csv");

    let out_dir = tmp.path().join("output");
    let output = Command::new(docling_bin())
        .args([
            "convert",
            bad_file.to_str().unwrap(),
            good_file.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
            "--to",
            "json",
        ])
        .output()
        .expect("failed to run");

    assert!(
        output.status.success(),
        "should succeed without abort-on-error; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        out_dir.join("csv-comma.json").exists(),
        "good file should still produce output"
    );
}

// --- --document-timeout ---

#[test]
fn test_cli_document_timeout_accepted() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let tmp = tempfile::TempDir::new().unwrap();

    let output = Command::new(docling_bin())
        .args([
            "convert",
            input.to_str().unwrap(),
            "-o",
            tmp.path().to_str().unwrap(),
            "--document-timeout",
            "60",
            "--to",
            "json",
        ])
        .output()
        .expect("failed to run");
    assert!(
        output.status.success(),
        "should accept --document-timeout; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// --- --num-threads ---

#[test]
fn test_cli_num_threads_accepted() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let tmp = tempfile::TempDir::new().unwrap();

    let output = Command::new(docling_bin())
        .args([
            "convert",
            input.to_str().unwrap(),
            "-o",
            tmp.path().to_str().unwrap(),
            "--num-threads",
            "2",
            "--to",
            "json",
        ])
        .output()
        .expect("failed to run");
    assert!(
        output.status.success(),
        "should accept --num-threads; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// --- --image-export-mode ---

#[test]
fn test_cli_image_export_mode_placeholder() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let tmp = tempfile::TempDir::new().unwrap();

    let output = Command::new(docling_bin())
        .args([
            "convert",
            input.to_str().unwrap(),
            "-o",
            tmp.path().to_str().unwrap(),
            "--image-export-mode",
            "placeholder",
            "--to",
            "md",
        ])
        .output()
        .expect("failed to run");
    assert!(
        output.status.success(),
        "should accept --image-export-mode; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// --- Directory input ---

#[test]
fn test_cli_directory_input() {
    let csv_dir = test_data_dir().join("csv");
    if csv_dir.is_dir() {
        let tmp = tempfile::TempDir::new().unwrap();

        let output = Command::new(docling_bin())
            .args([
                "convert",
                csv_dir.to_str().unwrap(),
                "-o",
                tmp.path().to_str().unwrap(),
                "--to",
                "json",
            ])
            .output()
            .expect("failed to run");
        assert!(
            output.status.success(),
            "directory input should work; stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let json_files: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        assert!(
            !json_files.is_empty(),
            "should produce at least one JSON output from directory"
        );
    }
}

// --- Output directory creation ---

#[test]
fn test_cli_output_dir_creation() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let tmp = tempfile::TempDir::new().unwrap();
    let nested_out = tmp.path().join("deep").join("nested").join("output");

    let output = Command::new(docling_bin())
        .args([
            "convert",
            input.to_str().unwrap(),
            "-o",
            nested_out.to_str().unwrap(),
            "--to",
            "json",
        ])
        .output()
        .expect("failed to run");
    assert!(
        output.status.success(),
        "should create nested output directory; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        nested_out.join("csv-comma.json").exists(),
        "output should be written to created directory"
    );
}

// --- Verbosity ---

#[test]
fn test_cli_verbose_flag() {
    let input = test_data_dir().join("csv").join("csv-comma.csv");
    let tmp = tempfile::TempDir::new().unwrap();

    let output = Command::new(docling_bin())
        .args([
            "convert",
            input.to_str().unwrap(),
            "-o",
            tmp.path().to_str().unwrap(),
            "-v",
            "--to",
            "json",
        ])
        .output()
        .expect("failed to run");
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Converting") || stderr.contains("Wrote"),
        "verbose mode should log info messages"
    );
}
