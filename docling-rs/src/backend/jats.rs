use std::path::Path;

use roxmltree::Document as XmlDoc;

use crate::models::common::{DocItemLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::table::TableCell;

use super::Backend;

pub struct JatsBackend;

impl Backend for JatsBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::XmlJats)?;
        let content = std::fs::read_to_string(path)?;
        let content = strip_dtd(&content);
        let xml = XmlDoc::parse(&content)?;

        let root = xml.root_element();

        // Extract title from front matter
        if let Some(front) = root.children().find(|n| n.has_tag_name("front")) {
            extract_front(&front, &mut doc);
        }

        // Extract body
        if let Some(body) = root.children().find(|n| n.has_tag_name("body")) {
            extract_body(&body, &mut doc, None);
        }

        // Extract back matter
        if let Some(back) = root.children().find(|n| n.has_tag_name("back")) {
            extract_back(&back, &mut doc);
        }

        Ok(doc)
    }
}

fn extract_front(node: &roxmltree::Node, doc: &mut DoclingDocument) {
    let mut title_found = false;
    for child in node.descendants() {
        if !title_found && child.has_tag_name("article-title") {
            let text = collect_text(child);
            if !text.is_empty() {
                doc.add_title(&text, None);
                title_found = true;
            }
        }
    }

    for child in node.descendants() {
        if child.has_tag_name("abstract") {
            extract_abstract(&child, doc);
        }
    }
}

fn extract_abstract(node: &roxmltree::Node, doc: &mut DoclingDocument) {
    let idx = doc.add_section_header("Abstract", 1, None);
    let parent = format!("#/texts/{}", idx);
    let parent_ref = Some(parent.as_str());

    let has_sections = node.children().any(|n| n.has_tag_name("sec"));
    if has_sections {
        for child in node.children() {
            if child.has_tag_name("sec") {
                extract_section(&child, doc, parent_ref, 2);
            }
        }
    } else {
        for child in node.children() {
            if child.has_tag_name("p") {
                let text = collect_text(child);
                if !text.is_empty() {
                    doc.add_text(DocItemLabel::Text, &text, parent_ref);
                }
            }
        }
    }
}

fn extract_body(
    node: &roxmltree::Node,
    doc: &mut DoclingDocument,
    parent_ref: Option<&str>,
) {
    for child in node.children() {
        if child.has_tag_name("sec") {
            extract_section(&child, doc, parent_ref, 1);
        } else if child.has_tag_name("p") {
            let text = collect_text(child);
            if !text.is_empty() {
                doc.add_text(DocItemLabel::Text, &text, parent_ref);
            }
        } else if child.has_tag_name("table-wrap") {
            extract_table(&child, doc, parent_ref);
        } else if child.has_tag_name("list") {
            extract_list(&child, doc, parent_ref);
        } else if child.has_tag_name("fig") {
            // Skip figures for now
        }
    }
}

fn extract_section(
    node: &roxmltree::Node,
    doc: &mut DoclingDocument,
    parent_ref: Option<&str>,
    depth: u32,
) {
    let mut section_ref: Option<String> = None;

    let label_text = node
        .children()
        .find(|n| n.has_tag_name("label"))
        .map(|n| collect_text(n))
        .unwrap_or_default();
    let title_text = node
        .children()
        .find(|n| n.has_tag_name("title"))
        .map(|n| collect_text(n))
        .unwrap_or_default();

    let header = match (label_text.is_empty(), title_text.is_empty()) {
        (false, false) => format!("{} {}", label_text, title_text),
        (false, true) => label_text,
        (true, false) => title_text,
        (true, true) => String::new(),
    };
    if !header.is_empty() {
        let idx = doc.add_section_header(&header, depth, parent_ref);
        section_ref = Some(format!("#/texts/{}", idx));
    }

    for child in node.children() {
        if child.has_tag_name("title") || child.has_tag_name("label") {
            continue;
        } else if child.has_tag_name("p") {
            let text = collect_text(child);
            if !text.is_empty() {
                doc.add_text(DocItemLabel::Text, &text, section_ref.as_deref().or(parent_ref));
            }
        } else if child.has_tag_name("sec") {
            extract_section(&child, doc, section_ref.as_deref().or(parent_ref), depth + 1);
        } else if child.has_tag_name("table-wrap") {
            extract_table(&child, doc, section_ref.as_deref().or(parent_ref));
        } else if child.has_tag_name("list") {
            extract_list(&child, doc, section_ref.as_deref().or(parent_ref));
        } else if child.has_tag_name("disp-formula") {
            let text = collect_text(child);
            if !text.is_empty() {
                doc.add_text(DocItemLabel::Formula, &text, section_ref.as_deref().or(parent_ref));
            }
        } else if child.has_tag_name("code") || child.has_tag_name("preformat") {
            let text = collect_text(child);
            if !text.is_empty() {
                doc.add_text(DocItemLabel::Code, &text, section_ref.as_deref().or(parent_ref));
            }
        } else if child.has_tag_name("fn-group") {
            for fn_node in child.children().filter(|n| n.has_tag_name("fn")) {
                let text = collect_text(fn_node);
                if !text.is_empty() {
                    doc.add_text(DocItemLabel::Footnote, &text, section_ref.as_deref().or(parent_ref));
                }
            }
        }
    }
}

fn extract_list(node: &roxmltree::Node, doc: &mut DoclingDocument, parent_ref: Option<&str>) {
    let is_ordered = node.attribute("list-type").map_or(false, |t| t == "order");
    for item in node.children().filter(|n| n.has_tag_name("list-item")) {
        let text = collect_text(item);
        if !text.is_empty() {
            if let Some(pr) = parent_ref {
                doc.add_list_item(&text, is_ordered, None, pr);
            } else {
                doc.add_text(DocItemLabel::ListItem, &text, None);
            }
        }
    }
}

fn extract_table(
    node: &roxmltree::Node,
    doc: &mut DoclingDocument,
    parent_ref: Option<&str>,
) {
    let label_text = node
        .children()
        .find(|n| n.has_tag_name("label"))
        .map(|n| collect_text(n))
        .unwrap_or_default();
    let caption_title = node
        .children()
        .find(|n| n.has_tag_name("caption"))
        .and_then(|cap| cap.children().find(|n| n.has_tag_name("title")))
        .map(|n| collect_text(n))
        .unwrap_or_default();
    let caption_text = match (label_text.is_empty(), caption_title.is_empty()) {
        (false, false) => Some(format!("{} {}", label_text, caption_title)),
        (false, true) => Some(label_text),
        (true, false) => Some(caption_title),
        (true, true) => None,
    };
    if let Some(ref cap) = caption_text {
        doc.add_text(DocItemLabel::Caption, cap, parent_ref);
    }

    let table_node = node
        .children()
        .find(|n| n.has_tag_name("table"))
        .unwrap_or(*node);

    let mut cells: Vec<TableCell> = Vec::new();
    let mut num_rows = 0u32;
    let mut num_cols = 0u32;

    // Track columns occupied by rowspans from previous rows.
    // Maps (row_idx, col_idx) to true if occupied.
    let mut occupied: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();

    for group in table_node.children() {
        if group.has_tag_name("thead") || group.has_tag_name("tbody") || group.has_tag_name("tfoot") || group.has_tag_name("tr") {
            let rows: Vec<roxmltree::Node> = if group.has_tag_name("tr") {
                vec![group]
            } else {
                group.children().filter(|n| n.has_tag_name("tr")).collect()
            };

            for row in rows {
                let mut col_idx = 0u32;
                let is_header = group.has_tag_name("thead");

                for cell_node in row.children() {
                    if cell_node.has_tag_name("th") || cell_node.has_tag_name("td") {
                        while occupied.contains(&(num_rows, col_idx)) {
                            col_idx += 1;
                        }

                        let text = collect_text(cell_node);
                        let colspan: u32 = cell_node
                            .attribute("colspan")
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(1);
                        let rowspan: u32 = cell_node
                            .attribute("rowspan")
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(1);

                        for r in 0..rowspan {
                            for c in 0..colspan {
                                if r > 0 || c > 0 {
                                    occupied.insert((num_rows + r, col_idx + c));
                                }
                            }
                        }

                        cells.push(TableCell {
                            row_span: rowspan,
                            col_span: colspan,
                            start_row_offset_idx: num_rows,
                            end_row_offset_idx: num_rows + rowspan,
                            start_col_offset_idx: col_idx,
                            end_col_offset_idx: col_idx + colspan,
                            text,
                            column_header: is_header || cell_node.has_tag_name("th"),
                            row_header: false,
                            row_section: false,
                            fillable: false,
                            formatted_text: None,
                        });
                        col_idx += colspan;
                    }
                }
                num_cols = num_cols.max(col_idx);
                num_rows += 1;
            }
        }
    }

    if !cells.is_empty() {
        doc.add_table(cells, num_rows, num_cols, parent_ref);
    }
}

fn extract_back(node: &roxmltree::Node, doc: &mut DoclingDocument) {
    for child in node.children() {
        if child.has_tag_name("ref-list") {
            let mut ref_section: Option<String> = None;
            if let Some(title) = child.children().find(|n| n.has_tag_name("title")) {
                let text = collect_text(title);
                if !text.is_empty() {
                    let idx = doc.add_section_header(&text, 1, None);
                    ref_section = Some(format!("#/texts/{}", idx));
                }
            }
            for ref_node in child.children().filter(|n| n.has_tag_name("ref")) {
                let text = collect_text(ref_node);
                if !text.is_empty() {
                    doc.add_text(DocItemLabel::Reference, &text, ref_section.as_deref());
                }
            }
        } else if child.has_tag_name("ack") {
            let idx = doc.add_section_header("Acknowledgments", 1, None);
            let parent = format!("#/texts/{}", idx);
            for p in child.descendants().filter(|n| n.has_tag_name("p")) {
                let text = collect_text(p);
                if !text.is_empty() {
                    doc.add_text(DocItemLabel::Text, &text, Some(&parent));
                }
            }
        } else if child.has_tag_name("app-group") {
            for app in child.children().filter(|n| n.has_tag_name("app")) {
                extract_section(&app, doc, None, 1);
            }
        } else if child.has_tag_name("sec") {
            extract_section(&child, doc, None, 1);
        } else if child.has_tag_name("fn-group") {
            for fn_node in child.children().filter(|n| n.has_tag_name("fn")) {
                let text = collect_text(fn_node);
                if !text.is_empty() {
                    doc.add_text(DocItemLabel::Footnote, &text, None);
                }
            }
        }
    }
}

fn strip_dtd(content: &str) -> String {
    let mut result = content.to_string();
    while let Some(start) = result.find("<!DOCTYPE") {
        let remaining = &result[start..];
        if let Some(bracket_pos) = remaining.find('[') {
            let first_gt = remaining.find('>').unwrap_or(usize::MAX);
            if bracket_pos < first_gt {
                if let Some(close_bracket) = remaining.find("]>") {
                    result = format!(
                        "{}{}",
                        &result[..start],
                        &result[start + close_bracket + 2..]
                    );
                    continue;
                }
            }
        }
        if let Some(end) = remaining.find('>') {
            result = format!("{}{}", &result[..start], &result[start + end + 1..]);
        } else {
            break;
        }
    }
    crate::backend::resolve_common_entities(&mut result);
    result
}

fn collect_text(node: roxmltree::Node) -> String {
    let mut result = String::new();
    collect_text_recursive(&node, &mut result);
    normalize_whitespace(&result)
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

fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            prev_ws = false;
            out.push(ch);
        }
    }
    out.trim().to_string()
}
