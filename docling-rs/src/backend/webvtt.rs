use std::path::Path;

use regex::Regex;

use crate::models::common::{DocItemLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};

use super::Backend;

pub struct WebVttBackend;

impl Backend for WebVttBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::Vtt)?;
        let content = std::fs::read_to_string(path)?;
        let content = content.strip_prefix('\u{FEFF}').unwrap_or(&content);

        let timestamp_re = Regex::new(
            r"(\d+:)?(\d{2}:\d{2}\.\d{3})\s*-->\s*(\d+:)?(\d{2}:\d{2}\.\d{3})",
        )
        .unwrap();
        let tag_re = Regex::new(r"</?[a-zA-Z][a-zA-Z0-9.]*(?:\s[^>]*)?>").unwrap();
        let rt_re = Regex::new(r"<rt>[^<]*</rt>").unwrap();

        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;

        // Skip WEBVTT header and any metadata lines that follow
        while i < lines.len() {
            let line = lines[i].trim();
            if line.starts_with("WEBVTT") {
                i += 1;
                while i < lines.len() && !lines[i].trim().is_empty() {
                    i += 1;
                }
                break;
            }
            i += 1;
        }

        while i < lines.len() {
            let line = lines[i].trim();
            if line.is_empty() {
                i += 1;
                continue;
            }

            // Explicitly skip NOTE blocks
            if line.starts_with("NOTE") {
                i += 1;
                while i < lines.len() && !lines[i].trim().is_empty() {
                    i += 1;
                }
                continue;
            }

            // Explicitly skip STYLE blocks
            if line.starts_with("STYLE") {
                i += 1;
                while i < lines.len() && !lines[i].trim().is_empty() {
                    i += 1;
                }
                continue;
            }

            // Explicitly skip REGION blocks
            if line.starts_with("REGION") {
                i += 1;
                while i < lines.len() && !lines[i].trim().is_empty() {
                    i += 1;
                }
                continue;
            }

            if timestamp_re.is_match(line) {
                i += 1;
            } else if i + 1 < lines.len() && timestamp_re.is_match(lines[i + 1].trim()) {
                i += 2;
            } else {
                i += 1;
                continue;
            }

            let mut cue_text = String::new();
            while i < lines.len() && !lines[i].trim().is_empty() {
                if !cue_text.is_empty() {
                    cue_text.push(' ');
                }
                let line_text = lines[i].trim();
                // Strip ruby annotation text before general tag removal
                let no_ruby = rt_re.replace_all(line_text, "");
                let cleaned = tag_re.replace_all(&no_ruby, "");
                cue_text.push_str(&cleaned);
                i += 1;
            }

            if !cue_text.is_empty() {
                doc.add_text(DocItemLabel::Text, &cue_text, None);
            }
        }

        Ok(doc)
    }
}
