use crate::models::common::{ContentLayer, DocItemLabel, GroupLabel, ImageRefMode};
use crate::models::document::DoclingDocument;
use crate::models::table::TableData;

pub fn export(doc: &DoclingDocument, image_mode: ImageRefMode) -> anyhow::Result<String> {
    let mut output = String::new();
    export_node(doc, "#/body", &mut output, image_mode, 0);
    let trimmed = output.trim_end_matches('\n');
    Ok(trimmed.to_string())
}

fn export_node(
    doc: &DoclingDocument,
    ref_path: &str,
    output: &mut String,
    image_mode: ImageRefMode,
    list_depth: u32,
) {
    if ref_path == "#/body" {
        for child in &doc.body.children {
            export_node(doc, &child.ref_path, output, image_mode, 0);
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/texts/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(text_item) = doc.texts.get(idx) {
                if text_item.content_layer == ContentLayer::Furniture {
                    return;
                }
                match text_item.label {
                    DocItemLabel::Title => {
                        if !output.is_empty() {
                            output.push('\n');
                        }
                        output.push_str("# ");
                        if let Some(ref url) = text_item.hyperlink {
                            output.push('[');
                            output.push_str(&text_item.text);
                            output.push_str("](");
                            output.push_str(url);
                            output.push(')');
                        } else {
                            output.push_str(&text_item.text);
                        }
                        output.push('\n');
                    }
                    DocItemLabel::SectionHeader => {
                        if !output.is_empty() {
                            output.push('\n');
                        }
                        let level = (text_item.level.unwrap_or(1) + 1).min(6);
                        for _ in 0..level {
                            output.push('#');
                        }
                        output.push(' ');
                        if let Some(ref url) = text_item.hyperlink {
                            output.push('[');
                            output.push_str(&text_item.text);
                            output.push_str("](");
                            output.push_str(url);
                            output.push(')');
                        } else {
                            output.push_str(&text_item.text);
                        }
                        output.push('\n');
                    }
                    DocItemLabel::Code => {
                        if !output.is_empty() {
                            output.push('\n');
                        }
                        output.push_str("```");
                        if let Some(ref lang) = text_item.code_language {
                            output.push_str(lang);
                        }
                        output.push('\n');
                        output.push_str(&text_item.text);
                        output.push('\n');
                        output.push_str("```\n");
                    }
                    DocItemLabel::Formula => {
                        if !output.is_empty() {
                            output.push('\n');
                        }
                        output.push_str("$$");
                        output.push_str(&text_item.text);
                        output.push_str("$$\n");
                    }
                    DocItemLabel::ListItem => {
                        let indent = "    ".repeat(list_depth.saturating_sub(1) as usize);
                        output.push_str(&indent);
                        let enumerated = text_item.enumerated.unwrap_or(false);
                        if enumerated {
                            let marker = text_item.marker.as_deref().unwrap_or("1.");
                            output.push_str(marker);
                            output.push(' ');
                        } else {
                            output.push_str("- ");
                        }
                        if let Some(ref url) = text_item.hyperlink {
                            output.push('[');
                            apply_formatting(
                                output,
                                &text_item.text,
                                text_item.formatting.as_ref(),
                            );
                            output.push_str("](");
                            output.push_str(url);
                            output.push(')');
                        } else {
                            apply_formatting(
                                output,
                                &text_item.text,
                                text_item.formatting.as_ref(),
                            );
                        }
                        output.push('\n');
                    }
                    DocItemLabel::Caption => {
                        if !output.is_empty() {
                            output.push('\n');
                        }
                        output.push('_');
                        output.push_str(&text_item.text);
                        output.push_str("_\n");
                    }
                    _ => {
                        if !text_item.text.is_empty() {
                            if !output.is_empty() {
                                output.push('\n');
                            }
                            if let Some(ref url) = text_item.hyperlink {
                                output.push('[');
                                apply_formatting(
                                    output,
                                    &text_item.text,
                                    text_item.formatting.as_ref(),
                                );
                                output.push_str("](");
                                output.push_str(url);
                                output.push_str(")\n");
                            } else {
                                apply_formatting(
                                    output,
                                    &text_item.text,
                                    text_item.formatting.as_ref(),
                                );
                                output.push('\n');
                            }
                        }
                    }
                }
                for child in &text_item.children {
                    export_node(doc, &child.ref_path, output, image_mode, list_depth);
                }
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/tables/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(table) = doc.tables.get(idx) {
                if !output.is_empty() {
                    output.push('\n');
                }
                output.push_str(&table_to_markdown(&table.data));
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
                let is_inline = matches!(
                    group.label,
                    GroupLabel::Inline
                );

                if is_inline {
                    export_inline_group(doc, group, output, image_mode);
                } else {
                    if is_list
                        && list_depth == 0
                        && !output.is_empty()
                        && !output.ends_with("\n\n")
                    {
                        output.push('\n');
                    }
                    let depth = if is_list { list_depth + 1 } else { list_depth };
                    for child in &group.children {
                        export_node(doc, &child.ref_path, output, image_mode, depth);
                    }
                }
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/pictures/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(pic) = doc.pictures.get(idx) {
                if !output.is_empty() {
                    output.push('\n');
                }
                let caption = pic
                    .meta
                    .as_ref()
                    .and_then(|m| m.description.as_deref())
                    .unwrap_or("");
                match image_mode {
                    ImageRefMode::Embedded | ImageRefMode::Referenced => {
                        if let Some(ref img) = pic.image {
                            // Check if image was omitted (filtered out)
                            if img.uri.starts_with("[IMAGE OMITTED") {
                                output.push_str(&img.uri);
                                output.push('\n');
                            } else {
                                // Extract image number from URI (e.g., "doc_images/image_0.jpg" -> 0)
                                let img_num = img.uri
                                    .rsplit('/')
                                    .next()
                                    .and_then(|f| f.strip_prefix("image_"))
                                    .and_then(|f| f.strip_suffix(".jpg"))
                                    .and_then(|n| n.parse::<usize>().ok())
                                    .map(|n| n + 1)  // 1-indexed for display
                                    .unwrap_or(idx + 1);
                                
                                let caption_part = if !caption.is_empty() {
                                    format!(": {}", caption)
                                } else {
                                    String::new()
                                };
                                output.push_str(&format!(
                                    "[IMAGE {}{} - see attached image {}]\n",
                                    img_num, caption_part, img_num
                                ));
                            }
                        } else {
                            output.push_str("<!-- image -->\n");
                        }
                    }
                    ImageRefMode::Placeholder => {
                        output.push_str("<!-- image -->\n");
                    }
                }
                for cap_ref in &pic.captions {
                    export_node(doc, &cap_ref.ref_path, output, image_mode, list_depth);
                }
                for child in &pic.children {
                    export_node(doc, &child.ref_path, output, image_mode, list_depth);
                }
            }
        }
        return;
    }
}

fn export_inline_group(
    doc: &DoclingDocument,
    group: &crate::models::document::GroupItem,
    output: &mut String,
    _image_mode: ImageRefMode,
) {
    if !output.is_empty() {
        output.push('\n');
    }
    for child in &group.children {
        if let Some(idx_str) = child.ref_path.strip_prefix("#/texts/") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if let Some(text_item) = doc.texts.get(idx) {
                    let text = &text_item.text;
                    if text.is_empty() {
                        continue;
                    }
                    if text_item.label == DocItemLabel::Formula {
                        output.push_str("$");
                        output.push_str(text);
                        output.push_str("$ ");
                    } else if let Some(ref url) = text_item.hyperlink {
                        output.push('[');
                        apply_formatting(output, text, text_item.formatting.as_ref());
                        output.push_str("](");
                        output.push_str(url);
                        output.push(')');
                        output.push(' ');
                    } else {
                        apply_formatting(output, text, text_item.formatting.as_ref());
                        output.push(' ');
                    }
                }
            }
        }
    }
    let len = output.len();
    if output.ends_with(' ') {
        output.truncate(len - 1);
    }
    output.push('\n');
}

fn apply_formatting(
    output: &mut String,
    text: &str,
    formatting: Option<&crate::models::text::TextFormatting>,
) {
    let mut prefix = String::new();
    let mut suffix = String::new();
    if let Some(fmt) = formatting {
        if fmt.bold {
            prefix.push_str("**");
            suffix.insert_str(0, "**");
        }
        if fmt.italic {
            prefix.push('*');
            suffix.insert(0, '*');
        }
        if fmt.strikethrough {
            prefix.push_str("~~");
            suffix.insert_str(0, "~~");
        }
    }
    output.push_str(&prefix);
    output.push_str(text);
    output.push_str(&suffix);
}

pub fn table_to_markdown(data: &TableData) -> String {
    if data.num_rows == 0 || data.num_cols == 0 {
        return String::new();
    }

    let nrows = data.num_rows as usize;
    let ncols = data.num_cols as usize;

    let mut flat_grid: Vec<Vec<String>> = vec![vec![String::new(); ncols]; nrows];

    let cells = if let Some(ref g) = data.grid {
        g.iter().flat_map(|row| row.iter()).collect::<Vec<_>>()
    } else {
        data.table_cells.iter().collect::<Vec<_>>()
    };

    for cell in &cells {
        let raw = cell.formatted_text.as_deref().unwrap_or(&cell.text);
        let text = escape_pipe(raw);
        let r0 = cell.start_row_offset_idx as usize;
        let c0 = cell.start_col_offset_idx as usize;
        let r1 = (r0 + cell.row_span as usize).min(nrows);
        let c1 = (c0 + cell.col_span as usize).min(ncols);
        for r in r0..r1 {
            for c in c0..c1 {
                flat_grid[r][c] = text.clone();
            }
        }
    }

    let is_numeric_col = detect_numeric_columns(&flat_grid);

    let mut col_widths = vec![0usize; ncols];
    for (row_idx, row) in flat_grid.iter().enumerate() {
        for (c, text) in row.iter().enumerate() {
            let len = text.chars().count();
            if row_idx == 0 {
                col_widths[c] = col_widths[c].max(len + 2);
            } else {
                col_widths[c] = col_widths[c].max(len);
            }
        }
    }

    let mut result = String::new();
    for (row_idx, row) in flat_grid.iter().enumerate() {
        result.push('|');
        for (col_idx, width) in col_widths.iter().enumerate() {
            let cell_text = &row[col_idx];
            let char_count = cell_text.chars().count();
            let pad = width.saturating_sub(char_count);
            if is_numeric_col[col_idx] {
                result.push(' ');
                for _ in 0..pad {
                    result.push(' ');
                }
                result.push_str(cell_text);
                result.push_str(" |");
            } else {
                result.push(' ');
                result.push_str(cell_text);
                for _ in 0..pad {
                    result.push(' ');
                }
                result.push_str(" |");
            }
        }
        result.push('\n');

        if row_idx == 0 {
            result.push('|');
            for width in &col_widths {
                result.push_str(&format!("{:-<width$}--|", "", width = *width));
            }
            result.push('\n');
        }
    }

    result
}

/// A column is numeric if every non-empty data cell (rows 1+) parses as a number.
fn detect_numeric_columns(grid: &[Vec<String>]) -> Vec<bool> {
    if grid.is_empty() {
        return Vec::new();
    }
    let ncols = grid[0].len();
    let mut is_numeric = vec![true; ncols];

    for (row_idx, row) in grid.iter().enumerate() {
        if row_idx == 0 {
            continue;
        }
        for (c, text) in row.iter().enumerate() {
            if !is_numeric[c] {
                continue;
            }
            let t = text.trim();
            if t.is_empty() {
                continue;
            }
            if t.parse::<f64>().is_err() {
                is_numeric[c] = false;
            }
        }
    }

    for c in 0..ncols {
        let has_data = grid.iter().skip(1).any(|row| !row[c].trim().is_empty());
        if !has_data {
            is_numeric[c] = false;
        }
    }

    is_numeric
}

fn escape_pipe(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', "<br>")
}
