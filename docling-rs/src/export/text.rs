use crate::models::document::DoclingDocument;
use crate::models::table::TableData;

pub fn export(doc: &DoclingDocument) -> anyhow::Result<String> {
    let mut output = String::new();
    export_node(doc, "#/body", &mut output);
    Ok(output)
}

fn export_node(doc: &DoclingDocument, ref_path: &str, output: &mut String) {
    if ref_path == "#/body" {
        for child in &doc.body.children {
            export_node(doc, &child.ref_path, output);
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/texts/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(text_item) = doc.texts.get(idx) {
                if !text_item.text.is_empty() {
                    if !output.is_empty() {
                        output.push('\n');
                    }
                    output.push_str(&text_item.text);
                    output.push('\n');
                }
                for child in &text_item.children {
                    export_node(doc, &child.ref_path, output);
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
                output.push_str(&table_to_plain_text(&table.data));
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

    if ref_path.starts_with("#/pictures/") {
        return;
    }
}

fn table_to_plain_text(data: &TableData) -> String {
    if let Some(ref grid) = data.grid {
        let mut rows: Vec<Vec<&str>> = Vec::new();
        for row_cells in grid {
            let row: Vec<&str> = row_cells.iter().map(|c| c.text.as_str()).collect();
            rows.push(row);
        }
        rows.iter()
            .map(|row| row.join("\t"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        let mut row_map: Vec<Vec<(u32, &str)>> = vec![Vec::new(); data.num_rows as usize];
        for cell in &data.table_cells {
            let r = cell.start_row_offset_idx as usize;
            if r < row_map.len() {
                row_map[r].push((cell.start_col_offset_idx, cell.text.as_str()));
            }
        }
        row_map
            .iter_mut()
            .map(|row| {
                row.sort_by_key(|(col, _)| *col);
                row.iter().map(|(_, txt)| *txt).collect::<Vec<_>>().join("\t")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
