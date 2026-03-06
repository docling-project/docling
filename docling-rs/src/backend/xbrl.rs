use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;
use roxmltree::Document as XmlDoc;

use crate::models::common::{DocItemLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::table::TableCell;

use super::Backend;

fn re_tag() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<[^>]+>").unwrap())
}
fn re_p() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?si)<p[^>]*>(.*?)</p>").unwrap())
}
fn re_bold_header() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^<b>(.+?)</b>$").unwrap())
}
fn re_tr() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?si)<tr[^>]*>(.*?)</tr>").unwrap())
}
fn re_cell() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?si)<(td|th)[^>]*>(.*?)</(?:td|th)>").unwrap())
}
fn re_colspan() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"colspan\s*=\s*["']?(\d+)"#).unwrap())
}
fn re_rowspan() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"rowspan\s*=\s*["']?(\d+)"#).unwrap())
}
fn re_hex_entity() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"&#x([0-9a-fA-F]+);").unwrap())
}
fn re_dec_entity() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"&#(\d+);").unwrap())
}

pub struct XbrlBackend;

impl Backend for XbrlBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::XmlXbrl)?;
        let content = std::fs::read_to_string(path)?;
        let content = strip_dtd(&content);
        let xml = XmlDoc::parse(&content)?;
        let root = xml.root_element();

        validate_xbrl_root(&root)?;

        let title = build_title(&root);
        if !title.is_empty() {
            doc.add_title(&title, None);
        }

        extract_text_blocks(&root, &mut doc);

        Ok(doc)
    }
}

fn validate_xbrl_root(root: &roxmltree::Node) -> anyhow::Result<()> {
    let ns = root.tag_name().namespace().unwrap_or("");
    let local = root.tag_name().name();
    if local == "xbrl"
        || ns.contains("xbrl.org")
        || ns.contains("xbrl.sec.gov")
        || local.ends_with("Filing")
    {
        return Ok(());
    }
    let has_text_block = root
        .children()
        .any(|c| c.is_element() && c.tag_name().name().ends_with("TextBlock"));
    if has_text_block {
        return Ok(());
    }
    anyhow::bail!(
        "XML does not appear to be an XBRL document (root element: <{}>)",
        local
    );
}

fn build_title(root: &roxmltree::Node) -> String {
    let mut doc_type = String::new();
    let mut registrant = String::new();
    let mut period_end = String::new();

    for child in root.children() {
        if !child.is_element() {
            continue;
        }
        let local = child.tag_name().name();
        match local {
            "DocumentType" => doc_type = collect_text(child).trim().to_string(),
            "EntityRegistrantName" => registrant = collect_text(child).trim().to_string(),
            "DocumentPeriodEndDate" => period_end = collect_text(child).trim().to_string(),
            _ => {}
        }
    }

    let mut parts = Vec::new();
    if !doc_type.is_empty() {
        parts.push(doc_type);
    }
    if !registrant.is_empty() {
        parts.push(registrant);
    }
    if !period_end.is_empty() {
        parts.push(period_end);
    }
    parts.join(" ")
}

fn extract_text_blocks(root: &roxmltree::Node, doc: &mut DoclingDocument) {
    for child in root.children() {
        if !child.is_element() {
            continue;
        }
        let local = child.tag_name().name();

        if local.ends_with("TextBlock") {
            let raw = collect_text(child);
            let html_content = unescape_html_entities(&raw);
            process_text_block(&html_content, doc);
        } else if is_dei_element(local) && !local.ends_with("TextBlock") {
            // Already handled in title building for key fields
        }
    }
}

fn is_dei_element(name: &str) -> bool {
    matches!(
        name,
        "DocumentType"
            | "DocumentAnnualReport"
            | "DocumentPeriodEndDate"
            | "CurrentFiscalYearEndDate"
            | "DocumentFiscalYearFocus"
            | "DocumentTransitionReport"
            | "EntityFileNumber"
            | "EntityRegistrantName"
            | "EntityIncorporationStateCountryCode"
            | "EntityTaxIdentificationNumber"
            | "EntityAddressAddressLine1"
            | "EntityAddressCityOrTown"
            | "EntityAddressStateOrProvince"
            | "EntityAddressPostalZipCode"
            | "CityAreaCode"
            | "LocalPhoneNumber"
            | "Security12bTitle"
            | "TradingSymbol"
            | "SecurityExchangeName"
    )
}

fn process_text_block(html: &str, doc: &mut DoclingDocument) {
    let table_ranges = find_top_level_tables(html);

    let mut last_end = 0;
    for (start, end) in &table_ranges {
        let before = &html[last_end..*start];
        extract_paragraphs_from_html(before, doc);

        parse_html_table(&html[*start..*end], doc);
        last_end = *end;
    }

    let remainder = &html[last_end..];
    extract_paragraphs_from_html(remainder, doc);
}

fn find_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let needle_lower: Vec<char> = needle.chars().map(|c| c.to_ascii_lowercase()).collect();
    let needle_len = needle_lower.len();
    let hay_chars: Vec<(usize, char)> = haystack.char_indices().collect();
    for i in 0..hay_chars.len() {
        if hay_chars.len() - i < needle_len {
            break;
        }
        let matches =
            (0..needle_len).all(|j| hay_chars[i + j].1.to_ascii_lowercase() == needle_lower[j]);
        if matches {
            return Some(hay_chars[i].0);
        }
    }
    None
}

fn find_top_level_tables(html: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut pos = 0;

    while pos < html.len() {
        let offset = match find_case_insensitive(&html[pos..], "<table") {
            Some(o) => o,
            None => break,
        };
        let table_start = pos + offset;

        let tag_close = match html[table_start..].find('>') {
            Some(c) => table_start + c + 1,
            None => {
                pos = table_start + 1;
                continue;
            }
        };

        let mut depth = 0;
        let mut scan = tag_close;
        let mut found_end = None;

        while scan < html.len() {
            let next_open = find_case_insensitive(&html[scan..], "<table");
            let next_close = find_case_insensitive(&html[scan..], "</table>");

            match (next_open, next_close) {
                (Some(o_off), Some(c_off)) if o_off < c_off => {
                    depth += 1;
                    let nested_close = html[scan + o_off..].find('>').unwrap_or(6);
                    scan += o_off + nested_close + 1;
                }
                (Some(_), Some(c_off)) | (None, Some(c_off)) => {
                    if depth == 0 {
                        found_end = Some(scan + c_off + "</table>".len());
                        break;
                    }
                    depth -= 1;
                    scan += c_off + "</table>".len();
                }
                (Some(o_off), None) => {
                    depth += 1;
                    let nested_close = html[scan + o_off..].find('>').unwrap_or(6);
                    scan += o_off + nested_close + 1;
                }
                (None, None) => break,
            }
        }

        if let Some(end) = found_end {
            ranges.push((table_start, end));
            pos = end;
        } else {
            pos = table_start + 1;
        }
    }
    ranges
}

fn extract_paragraphs_from_html(html: &str, doc: &mut DoclingDocument) {
    let tag_re = re_tag();
    let p_re = re_p();
    let bold_header_re = re_bold_header();

    let mut found_p = false;
    for cap in p_re.captures_iter(html) {
        found_p = true;
        let inner = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let inner_trimmed = inner.trim();

        if inner_trimmed.is_empty() || inner_trimmed == "&#160;" || inner_trimmed == "&nbsp;" {
            continue;
        }

        if let Some(bold_cap) = bold_header_re.captures(inner_trimmed) {
            let header_text = tag_re.replace_all(bold_cap.get(1).unwrap().as_str(), "");
            let header_text = decode_entities(&header_text).trim().to_string();
            if !header_text.is_empty() {
                if header_text.starts_with("NOTE ") || header_text.starts_with("Note ") {
                    doc.add_section_header(&header_text, 1, None);
                } else {
                    let clean = tag_re.replace_all(inner_trimmed, "");
                    let clean = decode_entities(&clean).trim().to_string();
                    if !clean.is_empty() {
                        doc.add_text(DocItemLabel::Text, &clean, None);
                    }
                }
                continue;
            }
        }

        let clean = tag_re.replace_all(inner_trimmed, " ");
        let clean = decode_entities(&clean);
        let clean = normalize_whitespace(&clean);
        if !clean.is_empty() {
            doc.add_text(DocItemLabel::Text, &clean, None);
        }
    }

    if !found_p {
        let clean = tag_re.replace_all(html, " ");
        let clean = decode_entities(&clean);
        let clean = normalize_whitespace(&clean);
        if !clean.is_empty() {
            doc.add_text(DocItemLabel::Text, &clean, None);
        }
    }
}

fn parse_html_table(table_html: &str, doc: &mut DoclingDocument) {
    let tag_re = re_tag();
    let tr_re = re_tr();
    let cell_re = re_cell();
    let colspan_re = re_colspan();
    let rowspan_re = re_rowspan();

    let mut cells: Vec<TableCell> = Vec::new();
    let mut num_rows = 0u32;
    let mut num_cols = 0u32;

    // Track cells occupied by rowspan from previous rows.
    // Key: (row, col), meaning that cell is occupied.
    let mut occupied: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();

    for tr_cap in tr_re.captures_iter(table_html) {
        let row_html = tr_cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let mut col_idx = 0u32;

        for cell_cap in cell_re.captures_iter(row_html) {
            // Skip columns occupied by earlier rowspans
            while occupied.contains(&(num_rows, col_idx)) {
                col_idx += 1;
            }

            let cell_tag = cell_cap.get(1).map(|m| m.as_str()).unwrap_or("td");
            let cell_content = cell_cap.get(2).map(|m| m.as_str()).unwrap_or("");
            let full_match = cell_cap.get(0).map(|m| m.as_str()).unwrap_or("");

            let colspan: u32 = colspan_re
                .captures(full_match)
                .and_then(|c| c.get(1))
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(1);
            let rowspan: u32 = rowspan_re
                .captures(full_match)
                .and_then(|c| c.get(1))
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(1);

            // Mark cells occupied by this rowspan/colspan
            if rowspan > 1 || colspan > 1 {
                for r in 0..rowspan {
                    for c in 0..colspan {
                        if r > 0 || c > 0 {
                            occupied.insert((num_rows + r, col_idx + c));
                        }
                    }
                }
            }

            let text = tag_re.replace_all(cell_content, " ");
            let text = decode_entities(&text);
            let text = normalize_whitespace(&text);

            cells.push(TableCell {
                row_span: rowspan,
                col_span: colspan,
                start_row_offset_idx: num_rows,
                end_row_offset_idx: num_rows + rowspan,
                start_col_offset_idx: col_idx,
                end_col_offset_idx: col_idx + colspan,
                text,
                column_header: cell_tag == "th",
                row_header: false,
                row_section: false,
                fillable: false,
                formatted_text: None,
            });
            col_idx += colspan;
        }
        // Also skip trailing occupied cells for accurate column count
        while occupied.contains(&(num_rows, col_idx)) {
            col_idx += 1;
        }
        num_cols = num_cols.max(col_idx);
        num_rows += 1;
    }

    if !cells.is_empty() {
        doc.add_table(cells, num_rows, num_cols, None);
    }
}

fn unescape_html_entities(s: &str) -> String {
    // First pass: restore HTML tags encoded in XBRL text blocks.
    // Only decode the structural entities needed to restore HTML.
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn decode_entities(s: &str) -> String {
    let re = re_hex_entity();
    let result = re.replace_all(s, |caps: &regex::Captures| {
        let hex = caps.get(1).unwrap().as_str();
        u32::from_str_radix(hex, 16)
            .ok()
            .and_then(char::from_u32)
            .filter(|c| *c != '\0')
            .map(|c| c.to_string())
            .unwrap_or_default()
    });
    let re2 = re_dec_entity();
    let result = re2.replace_all(&result, |caps: &regex::Captures| {
        let num: u32 = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
        char::from_u32(num)
            .filter(|c| *c != '\0')
            .map(|c| c.to_string())
            .unwrap_or_default()
    });
    result
        .replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace('\u{00a0}', " ")
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_dtd(content: &str) -> String {
    let mut result = content.to_string();
    let mut iterations = 0;
    while let Some(start) = result.find("<!DOCTYPE") {
        if let Some(end) = result[start..].find('>') {
            result = format!("{}{}", &result[..start], &result[start + end + 1..]);
        } else {
            break;
        }
        iterations += 1;
        if iterations > 100 {
            log::warn!("strip_dtd: too many iterations, aborting");
            break;
        }
    }
    result
}

fn collect_text(node: roxmltree::Node) -> String {
    let mut result = String::new();
    collect_text_recursive(&node, &mut result);
    result
}

fn collect_text_recursive(node: &roxmltree::Node, result: &mut String) {
    for child in node.children() {
        if child.is_text() {
            if let Some(text) = child.text() {
                result.push_str(text);
            }
        } else if child.is_element() {
            collect_text_recursive(&child, result);
        }
    }
}
