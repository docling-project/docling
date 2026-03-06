use crate::models::common::{DocItemLabel, GroupLabel, ImageRefMode};
use crate::models::document::DoclingDocument;

pub fn export(doc: &DoclingDocument, image_mode: ImageRefMode) -> anyhow::Result<String> {
    let mut output = String::from("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n");
    output.push_str(&format!("<title>{}</title>\n", html_escape(&doc.name)));
    output.push_str("</head>\n<body>\n");
    export_node(doc, "#/body", &mut output, image_mode);
    output.push_str("</body>\n</html>\n");
    Ok(output)
}

fn export_node(
    doc: &DoclingDocument,
    ref_path: &str,
    output: &mut String,
    image_mode: ImageRefMode,
) {
    if ref_path == "#/body" {
        for child in &doc.body.children {
            export_node(doc, &child.ref_path, output, image_mode);
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/texts/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(text_item) = doc.texts.get(idx) {
                let escaped = html_escape(&text_item.text);
                let formatted = apply_html_formatting(&escaped, text_item.formatting.as_ref());
                match text_item.label {
                    DocItemLabel::Title => {
                        if let Some(ref url) = text_item.hyperlink {
                            output.push_str(&format!(
                                "<h1><a href=\"{}\">{}</a></h1>\n",
                                html_escape(url),
                                formatted
                            ));
                        } else {
                            output.push_str(&format!("<h1>{}</h1>\n", formatted));
                        }
                    }
                    DocItemLabel::SectionHeader => {
                        let level = (text_item.level.unwrap_or(1) + 1).min(6);
                        if let Some(ref url) = text_item.hyperlink {
                            output.push_str(&format!(
                                "<h{l}><a href=\"{href}\">{text}</a></h{l}>\n",
                                l = level,
                                href = html_escape(url),
                                text = formatted,
                            ));
                        } else {
                            output.push_str(&format!("<h{}>{}</h{}>\n", level, formatted, level));
                        }
                    }
                    DocItemLabel::Code => {
                        let lang_attr = text_item
                            .code_language
                            .as_ref()
                            .map(|l| format!(" class=\"language-{}\"", html_escape(l)))
                            .unwrap_or_default();
                        output.push_str(&format!(
                            "<pre><code{}>{}</code></pre>\n",
                            lang_attr, escaped
                        ));
                    }
                    DocItemLabel::ListItem => {
                        if let Some(ref url) = text_item.hyperlink {
                            output.push_str(&format!(
                                "<li><a href=\"{}\">{}</a></li>\n",
                                html_escape(url),
                                formatted
                            ));
                        } else {
                            output.push_str(&format!("<li>{}</li>\n", formatted));
                        }
                    }
                    DocItemLabel::Formula => {
                        output.push_str(&format!(
                            "<div class=\"formula\">{}</div>\n",
                            escaped
                        ));
                    }
                    DocItemLabel::Caption => {
                        output.push_str(&format!("<figcaption>{}</figcaption>\n", formatted));
                    }
                    _ => {
                        if !text_item.text.is_empty() {
                            if let Some(ref url) = text_item.hyperlink {
                                output.push_str(&format!(
                                    "<p><a href=\"{}\">{}</a></p>\n",
                                    html_escape(url),
                                    formatted
                                ));
                            } else {
                                output.push_str(&format!("<p>{}</p>\n", formatted));
                            }
                        }
                    }
                }
                for child in &text_item.children {
                    export_node(doc, &child.ref_path, output, image_mode);
                }
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/tables/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(table) = doc.tables.get(idx) {
                output.push_str("<table>\n");
                let grid = table.data.grid.clone().unwrap_or_else(|| {
                    let mut g = vec![vec![]; table.data.num_rows as usize];
                    for cell in &table.data.table_cells {
                        let row = cell.start_row_offset_idx as usize;
                        if row < g.len() {
                            g[row].push(cell.clone());
                        }
                    }
                    for row in &mut g {
                        row.sort_by_key(|c| c.start_col_offset_idx);
                    }
                    g
                });

                let has_header = grid
                    .first()
                    .map(|row| row.iter().any(|c| c.column_header))
                    .unwrap_or(false);

                for (row_idx, row) in grid.iter().enumerate() {
                    if row_idx == 0 && has_header {
                        output.push_str("<thead>\n");
                    } else if (row_idx == 1 && has_header) || (row_idx == 0 && !has_header) {
                        output.push_str("<tbody>\n");
                    }

                    output.push_str("<tr>");
                    let tag = if row_idx == 0 && has_header {
                        "th"
                    } else {
                        "td"
                    };
                    for cell in row {
                        let mut attrs = String::new();
                        if cell.col_span > 1 {
                            attrs.push_str(&format!(" colspan=\"{}\"", cell.col_span));
                        }
                        if cell.row_span > 1 {
                            attrs.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
                        }
                        output.push_str(&format!(
                            "<{}{}>{}</{}>",
                            tag,
                            attrs,
                            html_escape(&cell.text),
                            tag
                        ));
                    }
                    output.push_str("</tr>\n");

                    if row_idx == 0 && has_header {
                        output.push_str("</thead>\n");
                    }
                }
                if has_header || !grid.is_empty() {
                    output.push_str("</tbody>\n");
                }
                output.push_str("</table>\n");
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/groups/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(group) = doc.groups.get(idx) {
                let is_list = matches!(
                    group.label,
                    GroupLabel::List | GroupLabel::OrderedList
                );
                let tag = if matches!(group.label, GroupLabel::OrderedList) {
                    "ol"
                } else {
                    "ul"
                };
                if is_list {
                    output.push_str(&format!("<{}>\n", tag));
                }
                for child in &group.children {
                    export_node(doc, &child.ref_path, output, image_mode);
                }
                if is_list {
                    output.push_str(&format!("</{}>\n", tag));
                }
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/pictures/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(pic) = doc.pictures.get(idx) {
                let alt = pic
                    .meta
                    .as_ref()
                    .and_then(|m| m.description.as_deref())
                    .unwrap_or("image");
                match image_mode {
                    ImageRefMode::Embedded | ImageRefMode::Referenced => {
                        if let Some(ref img) = pic.image {
                            output.push_str(&format!(
                                "<figure><img src=\"{}\" alt=\"{}\"/></figure>\n",
                                html_escape(&img.uri),
                                html_escape(alt)
                            ));
                        } else {
                            output.push_str("<figure><!-- image --></figure>\n");
                        }
                    }
                    ImageRefMode::Placeholder => {
                        output.push_str("<figure><!-- image --></figure>\n");
                    }
                }
            } else {
                log::debug!("HTML: picture index {} out of bounds", idx);
            }
        }
    }
}

fn apply_html_formatting(
    text: &str,
    formatting: Option<&crate::models::text::TextFormatting>,
) -> String {
    let mut result = text.to_string();
    if let Some(fmt) = formatting {
        if fmt.bold {
            result = format!("<strong>{}</strong>", result);
        }
        if fmt.italic {
            result = format!("<em>{}</em>", result);
        }
        if fmt.underline {
            result = format!("<u>{}</u>", result);
        }
        if fmt.strikethrough {
            result = format!("<s>{}</s>", result);
        }
        if let Some(ref script) = fmt.script {
            match script.as_str() {
                "superscript" => result = format!("<sup>{}</sup>", result),
                "subscript" => result = format!("<sub>{}</sub>", result),
                _ => {}
            }
        }
    }
    result
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
