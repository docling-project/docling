use std::path::Path;

use crate::models::common::InputFormat;
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::table::TableCell;

use super::Backend;

const UTF8_BOM: &str = "\u{FEFF}";

pub struct CsvBackend;

impl Backend for CsvBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::Csv)?;
        let raw_bytes = std::fs::read(path)?;
        let mut content = String::from_utf8(raw_bytes)
            .map_err(|_| anyhow::anyhow!("CSV file is not valid UTF-8"))?;

        let trimmed = content.trim();
        if trimmed.is_empty() {
            anyhow::bail!("CSV file is empty");
        }

        if content.starts_with(UTF8_BOM) {
            content = content[UTF8_BOM.len()..].to_string();
        }

        let delimiter = detect_delimiter(&content);
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .has_headers(true)
            .flexible(true)
            .from_reader(content.as_bytes());

        let headers = reader.headers()?.clone();
        let header_len = headers.len();

        let header_fields: Vec<String> = headers.iter().map(|h| h.to_string()).collect();

        let mut data_rows: Vec<Vec<String>> = Vec::new();
        let mut max_cols = header_len;
        for record in reader.records() {
            let record = record?;
            let fields: Vec<String> = record.iter().map(|f| f.to_string()).collect();
            max_cols = max_cols.max(fields.len());
            data_rows.push(fields);
        }

        let num_cols = max_cols as u32;
        let num_rows = (data_rows.len() + 1) as u32;
        let mut cells: Vec<TableCell> = Vec::new();

        for (col_idx, text) in header_fields.iter().enumerate() {
            cells.push(TableCell {
                row_span: 1,
                col_span: 1,
                start_row_offset_idx: 0,
                end_row_offset_idx: 1,
                start_col_offset_idx: col_idx as u32,
                end_col_offset_idx: (col_idx + 1) as u32,
                text: text.clone(),
                column_header: true,
                row_header: false,
                row_section: false,
                fillable: false,
                formatted_text: None,
            });
        }

        for (row_offset, fields) in data_rows.iter().enumerate() {
            let row_idx = (row_offset + 1) as u32;
            for (col_idx, field) in fields.iter().enumerate() {
                cells.push(TableCell {
                    row_span: 1,
                    col_span: 1,
                    start_row_offset_idx: row_idx,
                    end_row_offset_idx: row_idx + 1,
                    start_col_offset_idx: col_idx as u32,
                    end_col_offset_idx: (col_idx + 1) as u32,
                    text: field.clone(),
                    column_header: false,
                    row_header: false,
                    row_section: false,
                    fillable: false,
                    formatted_text: None,
                });
            }
        }

        doc.add_table(cells, num_rows, num_cols, None);
        Ok(doc)
    }
}

/// Count occurrences of `ch` outside quoted regions (RFC 4180 double-quote rules).
fn count_unquoted(line: &str, ch: char) -> usize {
    let mut count = 0;
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            if in_quotes {
                if chars.peek() == Some(&'"') {
                    chars.next(); // escaped quote
                } else {
                    in_quotes = false;
                }
            } else {
                in_quotes = true;
            }
        } else if !in_quotes && c == ch {
            count += 1;
        }
    }
    count
}

fn detect_delimiter(content: &str) -> u8 {
    let first_line = content.lines().next().unwrap_or("");

    let candidates: &[(u8, char)] = &[
        (b'\t', '\t'),
        (b'|', '|'),
        (b';', ';'),
        (b',', ','),
    ];

    let mut best = b',';
    let mut best_count = 0;

    for &(byte, ch) in candidates {
        let count = count_unquoted(first_line, ch);
        if count > best_count {
            best_count = count;
            best = byte;
        }
    }

    best
}
