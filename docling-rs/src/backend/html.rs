use std::collections::HashMap;
use std::path::Path;

use ego_tree::NodeRef;
use scraper::{Html, Node, Selector};

use crate::models::common::{DocItemLabel, GroupLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::picture::RefItem;
use crate::models::table::TableCell;

use super::Backend;

pub struct HtmlBackend;

impl Backend for HtmlBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::Html)?;
        let content = std::fs::read_to_string(path)?;
        let html = Html::parse_document(&content);

        let body_sel = Selector::parse("body").unwrap();
        let body = html.select(&body_sel).next();

        let root_id = body.map(|b| b.id()).unwrap_or(html.tree.root().id());
        let root_node = html.tree.get(root_id).unwrap();

        let mut ctx = HtmlCtx {
            doc: &mut doc,
            current_parent: None,
            list_parent: None,
        };

        process_element_children(&root_node, &mut ctx);
        Ok(doc)
    }
}

/// Parse an HTML fragment and add its content into an existing document.
pub fn parse_html_fragment(
    html_content: &str,
    doc: &mut DoclingDocument,
    parent_ref: Option<&str>,
) {
    let html = Html::parse_fragment(html_content);
    let root_node = html.tree.root();

    let mut ctx = HtmlCtx {
        doc,
        current_parent: parent_ref.map(|s| s.to_string()),
        list_parent: None,
    };

    process_element_children(&root_node, &mut ctx);
}

struct HtmlCtx<'a> {
    doc: &'a mut DoclingDocument,
    current_parent: Option<String>,
    list_parent: Option<String>,
}

fn process_element_children(node: &NodeRef<Node>, ctx: &mut HtmlCtx) {
    for child in node.children() {
        process_html_node(&child, ctx);
    }
}

fn process_html_node(node: &NodeRef<Node>, ctx: &mut HtmlCtx) {
    match node.value() {
        Node::Element(el) => {
            let tag = el.name.local.as_ref();
            match tag {
                "h1" => {
                    let text = collect_element_text(node);
                    if !text.is_empty() {
                        let idx = ctx.doc.add_title(&text, None);
                        if let Some(href) = extract_sole_link_href(node) {
                            ctx.doc.texts[idx].hyperlink = Some(href);
                        }
                        ctx.current_parent = Some(format!("#/texts/{}", idx));
                    }
                }
                "h2" | "h3" | "h4" | "h5" | "h6" => {
                    let level: u32 = tag[1..].parse().unwrap_or(2) - 1;
                    let text = collect_element_text(node);
                    if !text.is_empty() {
                        let idx = ctx.doc.add_section_header(&text, level, None);
                        if let Some(href) = extract_sole_link_href(node) {
                            ctx.doc.texts[idx].hyperlink = Some(href);
                        }
                        ctx.current_parent = Some(format!("#/texts/{}", idx));
                    }
                }
                "p" => {
                    let spans = collect_inline_spans(node);
                    if spans.len() == 1 && spans[0].href.is_none() {
                        let text = &spans[0].text;
                        if !text.is_empty() {
                            ctx.doc.add_text(
                                DocItemLabel::Text,
                                text,
                                ctx.current_parent.as_deref(),
                            );
                        }
                    } else {
                        for span in &spans {
                            if span.text.is_empty() {
                                continue;
                            }
                            let idx = ctx.doc.add_text(
                                DocItemLabel::Text,
                                &span.text,
                                ctx.current_parent.as_deref(),
                            );
                            if let Some(ref url) = span.href {
                                ctx.doc.texts[idx].hyperlink = Some(url.clone());
                            }
                        }
                    }
                }
                "pre" => {
                    let text = collect_preformatted_text(node);
                    if !text.is_empty() {
                        let idx = ctx.doc
                            .add_text(DocItemLabel::Code, &text, ctx.current_parent.as_deref());
                        if let Some(lang) = detect_code_language(node) {
                            ctx.doc.texts[idx].code_language = Some(lang);
                        }
                    }
                }
                "ul" => {
                    let parent = ctx.list_parent.clone().or(ctx.current_parent.clone());
                    let group_idx = ctx.doc.add_group(
                        "list",
                        GroupLabel::List,
                        parent.as_deref(),
                    );
                    let group_ref = format!("#/groups/{}", group_idx);
                    process_list_items(node, &group_ref, false, 1, ctx);
                }
                "ol" => {
                    let start: u32 = el
                        .attr("start")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1);
                    let parent = ctx.list_parent.clone().or(ctx.current_parent.clone());
                    let group_idx = ctx.doc.add_group(
                        "ordered list",
                        GroupLabel::OrderedList,
                        parent.as_deref(),
                    );
                    let group_ref = format!("#/groups/{}", group_idx);
                    process_list_items(node, &group_ref, true, start, ctx);
                }
                "dl" => {
                    let parent = ctx.list_parent.clone().or(ctx.current_parent.clone());
                    let group_idx = ctx.doc.add_group(
                        "list",
                        GroupLabel::List,
                        parent.as_deref(),
                    );
                    let group_ref = format!("#/groups/{}", group_idx);
                    for child in node.children() {
                        if let Node::Element(child_el) = child.value() {
                            let child_tag = child_el.name.local.as_ref();
                            if child_tag == "dt" || child_tag == "dd" {
                                let text = collect_element_text(&child);
                                if !text.is_empty() {
                                    ctx.doc.add_list_item(&text, false, Some("-"), &group_ref);
                                }
                            }
                        }
                    }
                }
                "table" => {
                    convert_table(node, ctx);
                }
                "a" => {
                    let text = collect_element_text(node);
                    if !text.is_empty() {
                        let href = el.attr("href").map(|s| s.to_string());
                        let idx = ctx.doc
                            .add_text(DocItemLabel::Text, &text, ctx.current_parent.as_deref());
                        if let Some(url) = href {
                            ctx.doc.texts[idx].hyperlink = Some(url);
                        }
                    }
                }
                "img" => {
                    let alt = el.attr("alt").unwrap_or("").to_string();
                    let alt_ref = if alt.is_empty() { None } else { Some(alt.as_str()) };
                    ctx.doc.add_picture(alt_ref, ctx.current_parent.as_deref());
                }
                "blockquote" => {
                    process_element_children(node, ctx);
                }
                "figure" => {
                    process_figure(node, ctx);
                }
                "figcaption" => {
                    let text = collect_element_text(node);
                    if !text.is_empty() {
                        ctx.doc.add_text(
                            DocItemLabel::Caption,
                            &text,
                            ctx.current_parent.as_deref(),
                        );
                    }
                }
                "div" | "section" | "article" | "main" | "header" | "aside"
                | "details" | "summary" | "span" => {
                    process_element_children(node, ctx);
                }
                "nav" => {
                    // Navigation elements are furniture, not main content
                }
                "footer" => {
                    let saved_parent = ctx.current_parent.clone();
                    ctx.current_parent = Some("#/furniture".to_string());
                    process_element_children(node, ctx);
                    ctx.current_parent = saved_parent;
                }
                "script" | "style" | "meta" | "link" | "noscript" | "head" | "title"
                | "template" | "svg" => {}
                _ => {
                    process_element_children(node, ctx);
                }
            }
        }
        Node::Text(t) => {
            let text = t.trim();
            if !text.is_empty() {
                ctx.doc
                    .add_text(DocItemLabel::Text, text, ctx.current_parent.as_deref());
            }
        }
        _ => {}
    }
}

fn process_figure(node: &NodeRef<Node>, ctx: &mut HtmlCtx) {
    let mut caption_text: Option<String> = None;
    let mut img_alt: Option<String> = None;

    for child in node.children() {
        if let Node::Element(el) = child.value() {
            let tag = el.name.local.as_ref();
            match tag {
                "figcaption" => {
                    let text = collect_element_text(&child);
                    if !text.is_empty() {
                        caption_text = Some(text);
                    }
                }
                "img" => {
                    img_alt = el.attr("alt").and_then(|a| {
                        let t = a.trim();
                        if t.is_empty() { None } else { Some(t.to_string()) }
                    });
                }
                _ => {}
            }
        }
    }

    let alt = img_alt.as_deref().or(caption_text.as_deref());
    let pic_idx = ctx.doc.add_picture(alt, ctx.current_parent.as_deref());

    if let Some(cap) = caption_text {
        let pic_ref = format!("#/pictures/{}", pic_idx);
        let cap_idx = ctx.doc.add_text(DocItemLabel::Caption, &cap, Some(&pic_ref));
        let cap_ref = RefItem {
            ref_path: format!("#/texts/{}", cap_idx),
        };
        ctx.doc.pictures[pic_idx].captions.push(cap_ref);
    }

    for child in node.children() {
        if let Node::Element(el) = child.value() {
            let tag = el.name.local.as_ref();
            if tag != "figcaption" && tag != "img" {
                process_html_node(&child, ctx);
            }
        }
    }
}

fn process_list_items(
    node: &NodeRef<Node>,
    group_ref: &str,
    ordered: bool,
    start: u32,
    ctx: &mut HtmlCtx,
) {
    let mut counter = start;
    for child in node.children() {
        if let Node::Element(child_el) = child.value() {
            if child_el.name.local.as_ref() == "li" {
                let text = collect_li_text(&child);
                if text.is_empty() {
                    continue;
                }
                let marker = if ordered {
                    format!("{}.", counter)
                } else {
                    "-".to_string()
                };
                ctx.doc
                    .add_list_item(&text, ordered, Some(&marker), group_ref);
                counter += 1;

                    // Handle nested lists inside <li>
                    let saved_list_parent = ctx.list_parent.clone();
                    ctx.list_parent = Some(group_ref.to_string());
                    for li_child in child.children() {
                        if let Node::Element(li_child_el) = li_child.value() {
                            let li_child_tag = li_child_el.name.local.as_ref();
                            if li_child_tag == "ul" || li_child_tag == "ol" {
                                process_html_node(&li_child, ctx);
                            }
                        }
                    }
                    ctx.list_parent = saved_list_parent;
            }
        }
    }
}

/// Collect only direct text from a <li>, excluding nested lists
fn collect_li_text(node: &NodeRef<Node>) -> String {
    let mut result = String::new();
    for child in node.children() {
        match child.value() {
            Node::Text(t) => result.push_str(t),
            Node::Element(el) => {
                let tag = el.name.local.as_ref();
                if tag != "ul" && tag != "ol" && tag != "dl" {
                    collect_text_recursive(&child, &mut result);
                }
            }
            _ => {}
        }
    }
    normalize_whitespace(result.trim())
}

fn convert_table(node: &NodeRef<Node>, ctx: &mut HtmlCtx) {
    let mut cells: Vec<TableCell> = Vec::new();
    let mut num_rows = 0u32;
    let mut num_cols = 0u32;

    let rows = collect_rows(node);
    if rows.is_empty() {
        return;
    }

    // First pass: determine dimensions and build occupancy grid
    let total_rows = rows.len();
    let mut occupied: HashMap<(usize, u32), bool> = HashMap::new();

    for (row_idx, row_node) in rows.iter().enumerate() {
        let mut col_idx: u32 = 0;
        let is_header_row = is_in_thead(row_node);

        for child in row_node.children() {
            if let Node::Element(el) = child.value() {
                let tag = el.name.local.as_ref();
                if tag == "th" || tag == "td" {
                    // Skip columns already occupied by rowspan from previous rows
                    while occupied.contains_key(&(row_idx, col_idx)) {
                        col_idx += 1;
                    }

                    let text = collect_element_text(&child);
                    let is_header = tag == "th" || is_header_row;
                    let col_span: u32 = el
                        .attr("colspan")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1);
                    let row_span: u32 = el
                        .attr("rowspan")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1);

                    // Mark cells as occupied for rowspan tracking
                    for dr in 0..row_span {
                        for dc in 0..col_span {
                            let r = row_idx + dr as usize;
                            if r < total_rows {
                                occupied.insert((r, col_idx + dc), true);
                            }
                        }
                    }

                    cells.push(TableCell {
                        row_span,
                        col_span,
                        start_row_offset_idx: row_idx as u32,
                        end_row_offset_idx: row_idx as u32 + row_span,
                        start_col_offset_idx: col_idx,
                        end_col_offset_idx: col_idx + col_span,
                        text,
                        column_header: is_header,
                        row_header: false,
                        row_section: false,
                        fillable: false,
                        formatted_text: None,
                    });
                    col_idx += col_span;
                }
            }
        }
        num_cols = num_cols.max(col_idx);
        num_rows += 1;
    }

    ctx.doc
        .add_table(cells, num_rows, num_cols, ctx.current_parent.as_deref());
}

fn collect_rows<'a>(table_node: &'a NodeRef<'a, Node>) -> Vec<NodeRef<'a, Node>> {
    let mut rows = Vec::new();
    for child in table_node.children() {
        if let Node::Element(el) = child.value() {
            let tag = el.name.local.as_ref();
            if tag == "tr" {
                rows.push(child);
            } else if tag == "thead" || tag == "tbody" || tag == "tfoot" {
                for sub in child.children() {
                    if let Node::Element(sub_el) = sub.value() {
                        if sub_el.name.local.as_ref() == "tr" {
                            rows.push(sub);
                        }
                    }
                }
            }
        }
    }
    rows
}

fn is_in_thead(node: &NodeRef<Node>) -> bool {
    if let Some(parent) = node.parent() {
        if let Node::Element(el) = parent.value() {
            return el.name.local.as_ref() == "thead";
        }
    }
    false
}

/// If the element contains a single <a> child with an href, return that href
fn extract_sole_link_href(node: &NodeRef<Node>) -> Option<String> {
    let mut link_href = None;
    let mut link_count = 0;
    for child in node.children() {
        if let Node::Element(el) = child.value() {
            if el.name.local.as_ref() == "a" {
                link_count += 1;
                link_href = el.attr("href").map(|s| s.to_string());
            }
        }
    }
    if link_count == 1 { link_href } else { None }
}

fn collect_element_text(node: &NodeRef<Node>) -> String {
    let mut result = String::new();
    collect_text_recursive(node, &mut result);
    normalize_whitespace(result.trim())
}

fn normalize_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for line in s.split('\n') {
        if !out.is_empty() {
            out.push('\n');
        }
        let mut prev_space = false;
        for ch in line.chars() {
            if ch.is_whitespace() {
                if !prev_space {
                    out.push(' ');
                    prev_space = true;
                }
            } else {
                out.push(ch);
                prev_space = false;
            }
        }
    }
    out
}

/// Preserve whitespace for <pre> elements
fn collect_preformatted_text(node: &NodeRef<Node>) -> String {
    let mut result = String::new();
    collect_preformatted_recursive(node, &mut result);
    // Only trim leading/trailing newlines, preserve internal whitespace
    result.trim_matches('\n').to_string()
}

fn collect_preformatted_recursive(node: &NodeRef<Node>, result: &mut String) {
    match node.value() {
        Node::Text(t) => {
            result.push_str(t);
        }
        Node::Element(el) => {
            let tag = el.name.local.as_ref();
            if tag == "br" {
                result.push('\n');
            } else {
                for child in node.children() {
                    collect_preformatted_recursive(&child, result);
                }
            }
        }
        _ => {}
    }
}

/// Detect code language from <pre><code class="language-X"> pattern
fn detect_code_language(pre_node: &NodeRef<Node>) -> Option<String> {
    for child in pre_node.children() {
        if let Node::Element(el) = child.value() {
            if el.name.local.as_ref() == "code" {
                if let Some(class) = el.attr("class") {
                    for cls in class.split_whitespace() {
                        if let Some(lang) = cls.strip_prefix("language-") {
                            return Some(lang.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

struct InlineSpan {
    text: String,
    href: Option<String>,
}

fn collect_inline_spans(node: &NodeRef<Node>) -> Vec<InlineSpan> {
    let mut spans = Vec::new();
    let mut current_text = String::new();

    for child in node.children() {
        match child.value() {
            Node::Text(t) => current_text.push_str(t),
            Node::Element(el) => {
                let tag = el.name.local.as_ref();
                if tag == "a" {
                    let pending = normalize_whitespace(current_text.trim());
                    if !pending.is_empty() {
                        spans.push(InlineSpan {
                            text: pending,
                            href: None,
                        });
                    }
                    current_text.clear();

                    let link_text = collect_element_text(&child);
                    let href = el.attr("href").map(|s| s.to_string());
                    if !link_text.is_empty() {
                        spans.push(InlineSpan {
                            text: link_text,
                            href,
                        });
                    }
                } else if tag == "br" {
                    current_text.push('\n');
                } else if tag == "img" {
                    // Skip images in paragraph inline spans
                } else {
                    collect_text_recursive(&child, &mut current_text);
                }
            }
            _ => {}
        }
    }

    let remaining = normalize_whitespace(current_text.trim());
    if !remaining.is_empty() {
        spans.push(InlineSpan {
            text: remaining,
            href: None,
        });
    }
    spans
}

fn collect_text_recursive(node: &NodeRef<Node>, result: &mut String) {
    match node.value() {
        Node::Text(t) => {
            result.push_str(t);
        }
        Node::Element(el) => {
            let tag = el.name.local.as_ref();
            if tag == "br" {
                result.push('\n');
            } else {
                for child in node.children() {
                    collect_text_recursive(&child, result);
                }
            }
        }
        _ => {}
    }
}
