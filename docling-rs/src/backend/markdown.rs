use std::path::Path;

use comrak::nodes::{ListType, NodeValue};
use comrak::{parse_document, Arena, Options};

use crate::models::common::{DocItemLabel, GroupLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::table::TableCell;

use super::Backend;

pub struct MarkdownBackend;

impl Backend for MarkdownBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::Md)?;
        let content = std::fs::read_to_string(path)?;

        let arena = Arena::new();
        let mut options = Options::default();
        options.extension.table = true;
        options.extension.strikethrough = true;
        options.extension.tasklist = true;
        options.extension.front_matter_delimiter = Some("---".to_string());

        let root = parse_document(&arena, &content, &options);
        let mut ctx = ConvertCtx {
            doc: &mut doc,
            current_parent: None,
            list_parent: None,
        };
        process_children(root, &mut ctx);
        Ok(doc)
    }
}

struct ConvertCtx<'a> {
    doc: &'a mut DoclingDocument,
    current_parent: Option<String>,
    list_parent: Option<String>,
}

fn process_children<'a>(node: &'a comrak::nodes::AstNode<'a>, ctx: &mut ConvertCtx) {
    for child in node.children() {
        process_node(child, ctx);
    }
}

fn process_node<'a>(node: &'a comrak::nodes::AstNode<'a>, ctx: &mut ConvertCtx) {
    let node_data = node.data.borrow();
    match &node_data.value {
        NodeValue::Heading(heading) => {
            let level = heading.level as u32;
            drop(node_data);
            let text = collect_text(node);
            if text.is_empty() {
                return;
            }
            if level == 1 {
                let idx = ctx.doc.add_title(&text, None);
                ctx.current_parent = Some(format!("#/texts/{}", idx));
            } else {
                let idx = ctx.doc.add_section_header(&text, level - 1, None);
                ctx.current_parent = Some(format!("#/texts/{}", idx));
            }
        }
        NodeValue::Paragraph => {
            drop(node_data);
            let spans = collect_paragraph_spans(node);
            for span in &spans {
                match span {
                    ParagraphSpan::Text { text, url } => {
                        if !text.is_empty() {
                            let idx = ctx.doc.add_text(
                                DocItemLabel::Text,
                                text,
                                ctx.current_parent.as_deref(),
                            );
                            if let Some(href) = url {
                                ctx.doc.texts[idx].hyperlink = Some(href.clone());
                            }
                        }
                    }
                    ParagraphSpan::Image { alt } => {
                        let alt_ref = if alt.is_empty() {
                            None
                        } else {
                            Some(alt.as_str())
                        };
                        ctx.doc.add_picture(alt_ref, ctx.current_parent.as_deref());
                    }
                }
            }
        }
        NodeValue::CodeBlock(cb) => {
            let text = cb.literal.trim_end().to_string();
            let info = cb.info.clone();
            drop(node_data);
            if text.is_empty() {
                return;
            }
            let idx = ctx.doc.add_text(
                DocItemLabel::Code,
                &text,
                ctx.current_parent.as_deref(),
            );
            if !info.is_empty() {
                let lang = info.split_whitespace().next().unwrap_or(&info);
                ctx.doc.texts[idx].code_language = Some(lang.to_string());
            }
        }
        NodeValue::List(list_node) => {
            let is_ordered = matches!(list_node.list_type, ListType::Ordered);
            let start = list_node.start as u32;
            drop(node_data);

            let label = if is_ordered {
                GroupLabel::OrderedList
            } else {
                GroupLabel::List
            };
            let name = if is_ordered {
                "ordered list"
            } else {
                "list"
            };
            let parent = ctx.list_parent.clone().or(ctx.current_parent.clone());
            let group_idx =
                ctx.doc
                    .add_group(name, label, parent.as_deref());
            let group_ref = format!("#/groups/{}", group_idx);

            let mut counter = start;
            for child in node.children() {
                let child_data = child.data.borrow();
                if let NodeValue::Item(_) = &child_data.value {
                    drop(child_data);
                    let item_text = collect_item_text(child);
                    let marker = if is_ordered {
                        let m = format!("{}.", counter);
                        counter += 1;
                        Some(m)
                    } else {
                        Some("-".to_string())
                    };
                    ctx.doc.add_list_item(
                        &item_text,
                        is_ordered,
                        marker.as_deref(),
                        &group_ref,
                    );

                    // Process nested lists within this item
                    let saved_list_parent = ctx.list_parent.clone();
                    ctx.list_parent = Some(group_ref.clone());
                    for item_child in child.children() {
                        let ic_data = item_child.data.borrow();
                        if matches!(&ic_data.value, NodeValue::List(_)) {
                            drop(ic_data);
                            process_node(item_child, ctx);
                        }
                    }
                    ctx.list_parent = saved_list_parent;
                } else if let NodeValue::TaskItem(checked) = &child_data.value {
                    let is_checked = checked.is_some();
                    drop(child_data);
                    let item_text = collect_item_text(child);
                    let prefix = if is_checked { "[x] " } else { "[ ] " };
                    let full_text = format!("{}{}", prefix, item_text);
                    ctx.doc.add_list_item(
                        &full_text,
                        false,
                        Some("-"),
                        &group_ref,
                    );

                    let saved_list_parent = ctx.list_parent.clone();
                    ctx.list_parent = Some(group_ref.clone());
                    for item_child in child.children() {
                        let ic_data = item_child.data.borrow();
                        if matches!(&ic_data.value, NodeValue::List(_)) {
                            drop(ic_data);
                            process_node(item_child, ctx);
                        }
                    }
                    ctx.list_parent = saved_list_parent;
                }
            }
        }
        NodeValue::Table(_) => {
            drop(node_data);
            let mut cells: Vec<TableCell> = Vec::new();
            let mut num_rows = 0u32;
            let mut num_cols = 0u32;

            for (row_idx, row_node) in node.children().enumerate() {
                let row_data = row_node.data.borrow();
                if !matches!(&row_data.value, NodeValue::TableRow(_)) {
                    continue;
                }
                drop(row_data);

                let mut col_idx = 0u32;
                for cell_node in row_node.children() {
                    let text = collect_text(cell_node);
                    cells.push(TableCell {
                        row_span: 1,
                        col_span: 1,
                        start_row_offset_idx: row_idx as u32,
                        end_row_offset_idx: (row_idx + 1) as u32,
                        start_col_offset_idx: col_idx,
                        end_col_offset_idx: col_idx + 1,
                        text,
                        column_header: row_idx == 0,
                        row_header: false,
                        row_section: false,
                        fillable: false,
                        formatted_text: None,
                    });
                    col_idx += 1;
                }
                num_cols = num_cols.max(col_idx);
                num_rows += 1;
            }

            ctx.doc
                .add_table(cells, num_rows, num_cols, ctx.current_parent.as_deref());
        }
        NodeValue::ThematicBreak | NodeValue::LineBreak | NodeValue::SoftBreak => {}
        NodeValue::FrontMatter(_) => {
            // Skip YAML frontmatter
        }
        NodeValue::HtmlBlock(hb) => {
            let html_content = hb.literal.trim().to_string();
            drop(node_data);
            if !html_content.is_empty() {
                super::html::parse_html_fragment(
                    &html_content,
                    ctx.doc,
                    ctx.current_parent.as_deref(),
                );
            }
        }
        NodeValue::BlockQuote => {
            drop(node_data);
            process_children(node, ctx);
        }
        _ => {
            drop(node_data);
            process_children(node, ctx);
        }
    }
}

enum ParagraphSpan {
    Text { text: String, url: Option<String> },
    Image { alt: String },
}

fn collect_paragraph_spans<'a>(node: &'a comrak::nodes::AstNode<'a>) -> Vec<ParagraphSpan> {
    let mut spans: Vec<ParagraphSpan> = Vec::new();
    let mut current_text = String::new();

    for child in node.children() {
        let child_data = child.data.borrow();
        match &child_data.value {
            NodeValue::Image(_) => {
                let trimmed = current_text.trim().to_string();
                if !trimmed.is_empty() {
                    spans.push(ParagraphSpan::Text {
                        text: trimmed,
                        url: None,
                    });
                }
                current_text.clear();
                drop(child_data);
                let alt = collect_text(child);
                spans.push(ParagraphSpan::Image { alt });
            }
            NodeValue::Link(link) => {
                let trimmed = current_text.trim().to_string();
                if !trimmed.is_empty() {
                    spans.push(ParagraphSpan::Text {
                        text: trimmed,
                        url: None,
                    });
                }
                current_text.clear();
                let url = link.url.clone();
                drop(child_data);
                let link_text = collect_text(child);
                if !link_text.is_empty() {
                    spans.push(ParagraphSpan::Text {
                        text: link_text,
                        url: Some(url),
                    });
                }
            }
            _ => {
                drop(child_data);
                collect_text_recursive(child, &mut current_text);
            }
        }
    }

    let trimmed = current_text.trim().to_string();
    if !trimmed.is_empty() {
        spans.push(ParagraphSpan::Text {
            text: trimmed,
            url: None,
        });
    }

    if spans.is_empty() {
        return spans;
    }

    // If there's only one plain text span, just return it directly
    if spans.len() == 1 {
        return spans;
    }

    spans
}

/// Collect text from a list item, excluding nested lists
fn collect_item_text<'a>(node: &'a comrak::nodes::AstNode<'a>) -> String {
    let mut result = String::new();
    for child in node.children() {
        let child_data = child.data.borrow();
        match &child_data.value {
            NodeValue::List(_) => {
                // Skip nested lists - they'll be processed separately
            }
            NodeValue::Paragraph => {
                drop(child_data);
                collect_text_recursive(child, &mut result);
            }
            _ => {
                drop(child_data);
                collect_text_recursive(child, &mut result);
            }
        }
    }
    result.trim().to_string()
}

fn collect_text<'a>(node: &'a comrak::nodes::AstNode<'a>) -> String {
    let mut result = String::new();
    collect_text_recursive(node, &mut result);
    result.trim().to_string()
}

fn collect_text_recursive<'a>(node: &'a comrak::nodes::AstNode<'a>, result: &mut String) {
    let data = node.data.borrow();
    match &data.value {
        NodeValue::Text(t) => result.push_str(t),
        NodeValue::Code(c) => {
            result.push_str(&c.literal);
        }
        NodeValue::SoftBreak | NodeValue::LineBreak => result.push(' '),
        NodeValue::HtmlInline(html) => {
            result.push_str(html);
        }
        NodeValue::FootnoteReference(r) => {
            result.push_str("[^");
            result.push_str(&r.name);
            result.push(']');
        }
        _ => {
            drop(data);
            for child in node.children() {
                collect_text_recursive(child, result);
            }
        }
    }
}
