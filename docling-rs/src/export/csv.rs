use crate::models::document::DoclingDocument;
use crate::models::table::TableData;

pub fn export(doc: &DoclingDocument) -> anyhow::Result<String> {
    let mut output = String::new();
    export_node(doc, "#/body", &mut output);
    let trimmed = output.trim_end_matches('\n');
    Ok(trimmed.to_string())
}

fn export_node(doc: &DoclingDocument, ref_path: &str, output: &mut String) {
    if ref_path == "#/body" {
        for child in &doc.body.children {
            export_node(doc, &child.ref_path, output);
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/tables/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(table) = doc.tables.get(idx) {
                if !output.is_empty() && !output.ends_with('\n') {
                    output.push('\n');
                }
                output.push_str(&table_to_csv(&table.data));
                output.push('\n');
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/groups/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(group) = doc.groups.get(idx) {
                for child in &group.children {
                    export_node(doc, &child.ref_path, output);
                }
            }
        }
        return;
    }
}

fn table_to_csv(data: &TableData) -> String {
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
        let r0 = cell.start_row_offset_idx as usize;
        let c0 = cell.start_col_offset_idx as usize;
        let r1 = (r0 + cell.row_span as usize).min(nrows);
        let c1 = (c0 + cell.col_span as usize).min(ncols);
        for r in r0..r1 {
            for c in c0..c1 {
                flat_grid[r][c] = cell.text.clone();
            }
        }
    }

    let mut result = String::new();
    for row in &flat_grid {
        for (i, cell) in row.iter().enumerate() {
            if i > 0 {
                result.push(',');
            }
            result.push_str(&csv_escape(cell));
        }
        result.push('\n');
    }
    result
}

/// RFC 4180 CSV field escaping: quote if the field contains commas,
/// double-quotes, or newlines; double any embedded quotes.
fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        let mut out = String::with_capacity(field.len() + 2);
        out.push('"');
        for ch in field.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
        out
    } else {
        field.to_string()
    }
}
