use std::path::Path;

use regex::Regex;

use crate::models::common::{DocItemLabel, GroupLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::table::TableCell;

use super::Backend;

pub struct AsciiDocBackend;

impl Backend for AsciiDocBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::AsciiDoc)?;
        let content = std::fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();

        let heading_re = Regex::new(r"^(={1,6})\s+(.+)$").unwrap();
        let ulist_re = Regex::new(r"^(\*+)\s+(.+)$").unwrap();
        let dash_list_re = Regex::new(r"^-\s+(.+)$").unwrap();
        let olist_re = Regex::new(r"^(\.{1,5})\s+(\S.*)$").unwrap();
        let table_delim_re = Regex::new(r"^\|===\s*$").unwrap();
        let listing_delim_re = Regex::new(r"^----+\s*$").unwrap();
        let literal_delim_re = Regex::new(r"^\.{4,}\s*$").unwrap();
        let comment_line_re = Regex::new(r"^//[^/]").unwrap();
        let comment_block_re = Regex::new(r"^/{4,}\s*$").unwrap();
        let inline_fmt_re = Regex::new(r"(?:\*([^*]+)\*|_([^_]+)_|`([^`]+)`)").unwrap();
        let admonition_re = Regex::new(r"^(NOTE|TIP|IMPORTANT|CAUTION|WARNING):\s+(.+)$").unwrap();
        let source_attr_re = Regex::new(r"^\[source(?:,\s*(\w+))?\]$").unwrap();
        let image_macro_re = Regex::new(r"^image::([^\[]+)\[([^\]]*)\]$").unwrap();

        let mut current_parent: Option<String> = None;
        let mut pending_source_lang: Option<String> = None;
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];

            if line.trim().is_empty() {
                pending_source_lang = None;
                i += 1;
                continue;
            }

            // Skip single-line comments
            if comment_line_re.is_match(line) {
                i += 1;
                continue;
            }

            // Skip comment blocks
            if comment_block_re.is_match(line) {
                i += 1;
                while i < lines.len() && !comment_block_re.is_match(lines[i]) {
                    i += 1;
                }
                if i < lines.len() {
                    i += 1;
                }
                continue;
            }

            // Source attribute: [source,lang]
            if let Some(caps) = source_attr_re.captures(line) {
                pending_source_lang = caps.get(1).map(|m| m.as_str().to_string());
                i += 1;
                continue;
            }

            // Image macro: image::path[alt]
            if let Some(caps) = image_macro_re.captures(line) {
                let alt = caps[2].trim().to_string();
                let alt_ref = if alt.is_empty() { None } else { Some(alt.as_str()) };
                doc.add_picture(alt_ref, current_parent.as_deref());
                i += 1;
                continue;
            }

            // Headings
            if let Some(caps) = heading_re.captures(line) {
                let level = caps[1].len() as u32;
                let text = strip_inline_formatting(&caps[2], &inline_fmt_re);
                let text = text.trim_end_matches(|c: char| c == '=' || c == ' ');
                if level == 1 {
                    let idx = doc.add_title(text, None);
                    current_parent = Some(format!("#/texts/{}", idx));
                } else {
                    let idx = doc.add_section_header(text, level - 1, None);
                    current_parent = Some(format!("#/texts/{}", idx));
                }
                i += 1;
                continue;
            }

            // Admonitions
            if let Some(caps) = admonition_re.captures(line) {
                let text = strip_inline_formatting(&caps[2], &inline_fmt_re);
                doc.add_text(DocItemLabel::Text, &text, current_parent.as_deref());
                i += 1;
                continue;
            }

            // Listing/source blocks (----)
            if listing_delim_re.is_match(line) {
                let lang = pending_source_lang.take();
                i += 1;
                let mut code_text = String::new();
                while i < lines.len() && !listing_delim_re.is_match(lines[i]) {
                    if !code_text.is_empty() {
                        code_text.push('\n');
                    }
                    code_text.push_str(lines[i]);
                    i += 1;
                }
                if i < lines.len() {
                    i += 1;
                }
                if !code_text.is_empty() {
                    let idx =
                        doc.add_text(DocItemLabel::Code, &code_text, current_parent.as_deref());
                    if let Some(l) = lang {
                        doc.texts[idx].code_language = Some(l);
                    }
                }
                continue;
            }

            // Literal blocks (....)
            if literal_delim_re.is_match(line) {
                pending_source_lang = None;
                i += 1;
                let mut code_text = String::new();
                while i < lines.len() && !literal_delim_re.is_match(lines[i]) {
                    if !code_text.is_empty() {
                        code_text.push('\n');
                    }
                    code_text.push_str(lines[i]);
                    i += 1;
                }
                if i < lines.len() {
                    i += 1;
                }
                if !code_text.is_empty() {
                    doc.add_text(DocItemLabel::Code, &code_text, current_parent.as_deref());
                }
                continue;
            }

            // Unordered lists (* or -)
            if ulist_re.is_match(line) || dash_list_re.is_match(line) {
                let group_idx =
                    doc.add_group("list", GroupLabel::List, current_parent.as_deref());
                let group_ref = format!("#/groups/{}", group_idx);
                while i < lines.len() {
                    if let Some(caps) = ulist_re.captures(lines[i]) {
                        let text = strip_inline_formatting(&caps[2], &inline_fmt_re);
                        doc.add_list_item(&text, false, Some("-"), &group_ref);
                        i += 1;
                    } else if let Some(caps) = dash_list_re.captures(lines[i]) {
                        let text = strip_inline_formatting(&caps[1], &inline_fmt_re);
                        doc.add_list_item(&text, false, Some("-"), &group_ref);
                        i += 1;
                    } else {
                        break;
                    }
                }
                continue;
            }

            // Ordered lists
            if olist_re.is_match(line) {
                let group_idx = doc.add_group(
                    "ordered list",
                    GroupLabel::OrderedList,
                    current_parent.as_deref(),
                );
                let group_ref = format!("#/groups/{}", group_idx);
                let mut counter = 1;
                while i < lines.len() {
                    if let Some(caps) = olist_re.captures(lines[i]) {
                        let text = strip_inline_formatting(&caps[2], &inline_fmt_re);
                        let marker = format!("{}.", counter);
                        doc.add_list_item(&text, true, Some(&marker), &group_ref);
                        counter += 1;
                        i += 1;
                    } else {
                        break;
                    }
                }
                continue;
            }

            // Tables
            if table_delim_re.is_match(line) {
                i += 1;
                let mut rows: Vec<Vec<String>> = Vec::new();
                let table_start = i;
                while i < lines.len() && !table_delim_re.is_match(lines[i]) {
                    let row_line = lines[i].trim();
                    if !row_line.is_empty() {
                        let cols = parse_table_row(row_line);
                        if !cols.is_empty() {
                            rows.push(cols);
                        }
                    }
                    i += 1;
                    // Guard: if we've consumed too many lines without closing,
                    // limit to 10000 lines to avoid eating the whole document
                    if i - table_start > 10_000 {
                        break;
                    }
                }
                if i < lines.len() {
                    i += 1; // skip closing |===
                }

                if !rows.is_empty() {
                    let num_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0) as u32;
                    let num_rows = rows.len() as u32;
                    let mut cells = Vec::new();
                    for (row_idx, row) in rows.iter().enumerate() {
                        for (col_idx, text) in row.iter().enumerate() {
                            let text = strip_inline_formatting(text, &inline_fmt_re);
                            cells.push(TableCell {
                                row_span: 1,
                                col_span: 1,
                                start_row_offset_idx: row_idx as u32,
                                end_row_offset_idx: (row_idx + 1) as u32,
                                start_col_offset_idx: col_idx as u32,
                                end_col_offset_idx: (col_idx + 1) as u32,
                                text,
                                column_header: row_idx == 0,
                                row_header: false,
                                row_section: false,
                                fillable: false,
                                formatted_text: None,
                            });
                        }
                    }
                    doc.add_table(cells, num_rows, num_cols, current_parent.as_deref());
                }
                continue;
            }

            // Skip attribute lines like :key: value
            if line.starts_with(':') && line.len() > 1 && line[1..].contains(':') {
                i += 1;
                continue;
            }

            // Regular paragraph - stop at any recognized structure
            let mut para_text = String::from(line.trim());
            i += 1;
            while i < lines.len()
                && !lines[i].trim().is_empty()
                && !heading_re.is_match(lines[i])
                && !ulist_re.is_match(lines[i])
                && !dash_list_re.is_match(lines[i])
                && !olist_re.is_match(lines[i])
                && !table_delim_re.is_match(lines[i])
                && !listing_delim_re.is_match(lines[i])
                && !literal_delim_re.is_match(lines[i])
                && !comment_line_re.is_match(lines[i])
                && !comment_block_re.is_match(lines[i])
                && !admonition_re.is_match(lines[i])
            {
                para_text.push(' ');
                para_text.push_str(lines[i].trim());
                i += 1;
            }

            let para_text = strip_inline_formatting(&para_text, &inline_fmt_re);
            doc.add_text(
                DocItemLabel::Text,
                &para_text,
                current_parent.as_deref(),
            );
        }

        Ok(doc)
    }
}

/// Parse a table row, handling escaped pipes
fn parse_table_row(line: &str) -> Vec<String> {
    let mut cols = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut in_content = false;

    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&'|') = chars.peek() {
                current.push('|');
                chars.next();
                continue;
            }
            current.push(c);
        } else if c == '|' {
            if in_content {
                cols.push(current.trim().to_string());
                current = String::new();
            }
            in_content = true;
        } else {
            current.push(c);
        }
    }
    // Include trailing content if there is any
    let remaining = current.trim().to_string();
    if !remaining.is_empty() && in_content {
        cols.push(remaining);
    }
    cols
}

/// Strip basic AsciiDoc inline formatting markers
fn strip_inline_formatting(text: &str, re: &Regex) -> String {
    let mut result = text.to_string();
    loop {
        let replaced = re.replace_all(&result, |caps: &regex::Captures| {
            if let Some(m) = caps.get(1) {
                m.as_str().to_string()
            } else if let Some(m) = caps.get(2) {
                m.as_str().to_string()
            } else if let Some(m) = caps.get(3) {
                m.as_str().to_string()
            } else {
                caps[0].to_string()
            }
        });
        if replaced == result {
            break;
        }
        result = replaced.to_string();
    }
    result
}
