use std::path::Path;

use roxmltree::Document as XmlDoc;

use crate::models::common::{DocItemLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::table::TableCell;

use super::Backend;

pub struct UsptoBackend;

impl Backend for UsptoBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::XmlUspto)?;
        let content = std::fs::read_to_string(path)?;

        let content = strip_dtd(&content);
        let looks_like_xml = content.trim_start().starts_with('<');

        if looks_like_xml {
            let xml = XmlDoc::parse(&content)
                .map_err(|e| anyhow::anyhow!("Failed to parse USPTO XML: {}", e))?;
            let root = xml.root_element();
            let tag = root.tag_name().name();

            match tag {
                "us-patent-application" | "us-patent-grant" | "patent-application-publication" => {
                    parse_modern_patent(&root, &mut doc);
                }
                _ => {
                    parse_generic_xml(&root, &mut doc);
                }
            }
        } else {
            parse_aps_text(&content, &mut doc);
        }

        Ok(doc)
    }
}

fn parse_modern_patent(root: &roxmltree::Node, doc: &mut DoclingDocument) {
    // Extract title
    for node in root.descendants() {
        if node.has_tag_name("invention-title") {
            let text = collect_text(node);
            if !text.is_empty() {
                doc.add_title(&text, None);
                break;
            }
        }
    }

    // Extract abstract (handles both <abstract> and <subdoc-abstract>)
    for node in root.descendants() {
        if node.has_tag_name("abstract") || node.has_tag_name("subdoc-abstract") {
            let idx = doc.add_section_header("Abstract", 1, None);
            let parent = format!("#/texts/{}", idx);
            for p in node
                .descendants()
                .filter(|n| n.has_tag_name("p") || n.has_tag_name("paragraph"))
            {
                let text = collect_text(p);
                if !text.is_empty() {
                    doc.add_text(DocItemLabel::Text, &text, Some(&parent));
                }
            }
            break;
        }
    }

    // Extract claims (handles both <claims>/<claim> and <subdoc-claims>)
    for node in root.descendants() {
        if node.has_tag_name("claims") || node.has_tag_name("subdoc-claims") {
            let idx = doc.add_section_header("Claims", 1, None);
            let parent = format!("#/texts/{}", idx);
            let has_claim_elements = node.descendants().any(|n| n.has_tag_name("claim"));
            if has_claim_elements {
                for claim in node.descendants().filter(|n| n.has_tag_name("claim")) {
                    let text = collect_text(claim);
                    if !text.is_empty() {
                        doc.add_text(DocItemLabel::Text, &text, Some(&parent));
                    }
                }
            } else {
                for p in node.descendants().filter(|n| {
                    n.has_tag_name("p")
                        || n.has_tag_name("paragraph")
                        || n.has_tag_name("claim-text")
                }) {
                    let text = collect_text(p);
                    if !text.is_empty() {
                        doc.add_text(DocItemLabel::Text, &text, Some(&parent));
                    }
                }
            }
            break;
        }
    }

    // Extract description (handles both <description> and <subdoc-description>)
    for node in root.descendants() {
        if node.has_tag_name("description") || node.has_tag_name("subdoc-description") {
            let idx = doc.add_section_header("Description", 1, None);
            let parent = format!("#/texts/{}", idx);
            extract_description_children(&node, doc, &parent);
            break;
        }
    }
}

fn extract_description_children(node: &roxmltree::Node, doc: &mut DoclingDocument, parent: &str) {
    for child in node.children() {
        if !child.is_element() {
            continue;
        }
        let tag = child.tag_name().name();
        if tag == "p" || tag == "paragraph" {
            let text = collect_text(child);
            if !text.is_empty() {
                doc.add_text(DocItemLabel::Text, &text, Some(parent));
            }
        } else if tag == "heading" {
            let text = collect_text(child);
            if !text.is_empty() {
                doc.add_section_header(&text, 2, Some(parent));
            }
        } else if tag == "table" {
            extract_table(&child, doc, Some(parent));
        } else if tag == "ul" || tag == "ol" {
            for li in child.children().filter(|n| n.has_tag_name("li")) {
                let text = collect_text(li);
                if !text.is_empty() {
                    doc.add_list_item(&text, false, None, parent);
                }
            }
        } else {
            let is_wrapper = matches!(
                tag,
                "description-of-drawings"
                    | "description-of-embodiments"
                    | "technical-field"
                    | "background-art"
                    | "disclosure"
                    | "description-of-preferred-embodiment"
                    | "summary-of-invention"
                    | "brief-description-of-drawings"
                    | "detailed-description"
                    | "cross-reference-to-related-applications"
                    | "federal-research-statement"
                    | "section"
            );
            if is_wrapper {
                extract_description_children(&child, doc, parent);
            }
        }
    }
}

fn extract_table(node: &roxmltree::Node, doc: &mut DoclingDocument, parent_ref: Option<&str>) {
    let mut cells: Vec<TableCell> = Vec::new();
    let mut num_rows = 0u32;
    let mut num_cols = 0u32;
    let mut occupied: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();

    for group in node.children() {
        let rows: Vec<roxmltree::Node> = if group.has_tag_name("tr") || group.has_tag_name("row") {
            vec![group]
        } else if group.has_tag_name("thead")
            || group.has_tag_name("tbody")
            || group.has_tag_name("tfoot")
            || group.has_tag_name("tgroup")
        {
            group
                .children()
                .filter(|n| n.has_tag_name("tr") || n.has_tag_name("row"))
                .collect()
        } else {
            continue;
        };

        for row in rows {
            let mut col_idx = 0u32;
            let is_header = group.has_tag_name("thead");

            for cell_node in row.children() {
                if cell_node.has_tag_name("th")
                    || cell_node.has_tag_name("td")
                    || cell_node.has_tag_name("entry")
                {
                    while occupied.contains(&(num_rows, col_idx)) {
                        col_idx += 1;
                    }

                    let text = collect_text(cell_node);
                    let colspan: u32 = cell_node
                        .attribute("colspan")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1);
                    let rowspan: u32 = if let Some(v) = cell_node.attribute("rowspan") {
                        v.parse().unwrap_or(1)
                    } else if let Some(v) = cell_node.attribute("morerows") {
                        v.parse::<u32>().map(|n| n + 1).unwrap_or(1)
                    } else {
                        1
                    };

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

    if !cells.is_empty() {
        doc.add_table(cells, num_rows, num_cols, parent_ref);
    }
}

fn parse_generic_xml(root: &roxmltree::Node, doc: &mut DoclingDocument) {
    // Try to extract title from known PATDOC elements
    for node in root.descendants() {
        let tag = node.tag_name().name();
        if tag == "B540" || tag == "b540" || tag == "invention-title" {
            let text = collect_text(node);
            if !text.is_empty() {
                doc.add_title(&text, None);
                break;
            }
        }
    }

    // Extract sections from PATDOC-style documents
    let section_tags = [
        ("SDOAB", "Abstract"),
        ("sdoab", "Abstract"),
        ("SDOCL", "Claims"),
        ("sdocl", "Claims"),
        ("SDODE", "Description"),
        ("sdode", "Description"),
    ];
    let mut found_sections = false;
    for (tag, label) in &section_tags {
        for node in root.descendants() {
            if node.tag_name().name() == *tag {
                found_sections = true;
                let idx = doc.add_section_header(label, 1, None);
                let parent = format!("#/texts/{}", idx);
                extract_patdoc_section(&node, doc, &parent);
            }
        }
    }

    if !found_sections {
        for node in root.descendants() {
            if node.is_element() {
                let has_children = node.children().any(|c| c.is_element());
                if !has_children {
                    let text = collect_text(node);
                    let text = text.trim();
                    if !text.is_empty() {
                        doc.add_text(DocItemLabel::Text, text, None);
                    }
                }
            }
        }
    }
}

fn extract_patdoc_section(node: &roxmltree::Node, doc: &mut DoclingDocument, parent: &str) {
    let mut current_parent = parent.to_string();
    for child in node.children() {
        if !child.is_element() {
            continue;
        }
        let tag = child.tag_name().name();
        if tag == "H" || tag == "h" {
            let text = collect_text(child);
            if !text.is_empty() {
                let idx = doc.add_section_header(&text, 2, Some(parent));
                current_parent = format!("#/texts/{}", idx);
            }
        } else if tag == "PARA"
            || tag == "para"
            || tag == "p"
            || tag == "paragraph"
            || tag == "CLMSTEP"
            || tag == "clmstep"
        {
            let has_nested_paras = child.children().any(|c| {
                let t = c.tag_name().name();
                matches!(t, "PARA" | "para" | "CLMSTEP" | "clmstep")
            });
            if has_nested_paras {
                extract_patdoc_section(&child, doc, &current_parent);
                continue;
            }
            let text = collect_text(child);
            if !text.is_empty() {
                doc.add_text(DocItemLabel::Text, &text, Some(&current_parent));
            }
        } else {
            extract_patdoc_section(&child, doc, &current_parent);
        }
    }
}

fn parse_aps_text(content: &str, doc: &mut DoclingDocument) {
    let mut current_section: Option<String> = None;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with("TTL ") || line.starts_with("TITL") {
            let text = line.split_once(' ').map(|(_, t)| t).unwrap_or(line);
            doc.add_title(text.trim(), None);
        } else if line.starts_with("ABST")
            || line.starts_with("BSUM")
            || line.starts_with("CLMS")
            || line.starts_with("DETD")
        {
            let section_name = line.split_whitespace().next().unwrap_or(line);
            let idx = doc.add_section_header(section_name, 1, None);
            current_section = Some(format!("#/texts/{}", idx));
        } else if line.starts_with("PAR ") || line.starts_with("PAL ") {
            let text = line.split_once(' ').map(|(_, t)| t).unwrap_or(line);
            if !text.trim().is_empty() {
                doc.add_text(DocItemLabel::Text, text.trim(), current_section.as_deref());
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
    super::resolve_common_entities(&mut result);
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
