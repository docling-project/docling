use serde::{Deserialize, Serialize};

use super::common::{ContentLayer, DocItemLabel};
use super::page::ProvenanceItem;
use super::picture::RefItem;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCell {
    #[serde(default = "default_one")]
    pub row_span: u32,
    #[serde(default = "default_one")]
    pub col_span: u32,
    #[serde(default)]
    pub start_row_offset_idx: u32,
    #[serde(default)]
    pub end_row_offset_idx: u32,
    #[serde(default)]
    pub start_col_offset_idx: u32,
    #[serde(default)]
    pub end_col_offset_idx: u32,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub column_header: bool,
    #[serde(default)]
    pub row_header: bool,
    #[serde(default)]
    pub row_section: bool,
    #[serde(default)]
    pub fillable: bool,
    #[serde(skip)]
    pub formatted_text: Option<String>,
}

fn default_one() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableData {
    pub table_cells: Vec<TableCell>,
    pub num_rows: u32,
    pub num_cols: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid: Option<Vec<Vec<TableCell>>>,
}

impl TableData {
    pub fn build_grid(&mut self) {
        let num_rows = self.num_rows as usize;
        let num_cols = self.num_cols as usize;

        let mut occupied = std::collections::HashSet::new();
        for cell in &self.table_cells {
            for r in cell.start_row_offset_idx..cell.end_row_offset_idx {
                for c in cell.start_col_offset_idx..cell.end_col_offset_idx {
                    occupied.insert((r, c));
                }
            }
        }

        let mut grid = vec![vec![]; num_rows];
        for cell in &self.table_cells {
            let row = cell.start_row_offset_idx as usize;
            if row < grid.len() {
                grid[row].push(cell.clone());
            }
        }
        for (row_idx, row) in grid.iter_mut().enumerate() {
            row.sort_by_key(|c| c.start_col_offset_idx);
            for col in 0..num_cols {
                if !occupied.contains(&(row_idx as u32, col as u32)) {
                    row.push(TableCell {
                        row_span: 1,
                        col_span: 1,
                        start_row_offset_idx: row_idx as u32,
                        end_row_offset_idx: (row_idx + 1) as u32,
                        start_col_offset_idx: col as u32,
                        end_col_offset_idx: (col + 1) as u32,
                        text: String::new(),
                        column_header: false,
                        row_header: false,
                        row_section: false,
                        fillable: false,
                        formatted_text: None,
                    });
                }
            }
            row.sort_by_key(|c| c.start_col_offset_idx);
        }
        self.grid = Some(grid);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableItem {
    pub self_ref: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<RefItem>,
    #[serde(default)]
    pub children: Vec<RefItem>,
    #[serde(default)]
    pub content_layer: ContentLayer,
    pub label: DocItemLabel,
    #[serde(default)]
    pub prov: Vec<ProvenanceItem>,
    #[serde(default)]
    pub captions: Vec<RefItem>,
    #[serde(default)]
    pub references: Vec<RefItem>,
    #[serde(default)]
    pub footnotes: Vec<RefItem>,
    pub data: TableData,
    #[serde(default)]
    pub annotations: Vec<serde_json::Value>,
}
