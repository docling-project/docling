use std::collections::{HashMap, VecDeque};
use std::io::Read as IoRead;
use std::path::Path;

use anyhow::Context;
use calamine::{open_workbook_auto, Data, Dimensions, Reader};

use crate::models::common::{GroupLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::table::TableCell;

use super::Backend;

/// Refuse to allocate occupancy grids larger than this to prevent OOM on
/// inflated or corrupt sheets.
const MAX_GRID_CELLS: usize = 50_000_000;

pub struct XlsxBackend;

impl Backend for XlsxBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::Xlsx)?;
        let mut workbook = open_workbook_auto(path)
            .with_context(|| format!("Failed to open spreadsheet: {}", path.display()))?;

        let sheet_names: Vec<String> = workbook.sheet_names().to_vec();

        let all_merges = load_all_merged_regions(&mut workbook, &sheet_names);
        let images_per_sheet = count_images_per_sheet(path, &sheet_names);

        for sheet_name in &sheet_names {
            let range = match workbook.worksheet_range(sheet_name) {
                Ok(r) => r,
                Err(_) => {
                    doc.add_group(&format!("sheet: {}", sheet_name), GroupLabel::Section, None);
                    continue;
                }
            };
            let (height, width) = range.get_size();
            if height == 0 || width == 0 {
                doc.add_group(&format!("sheet: {}", sheet_name), GroupLabel::Section, None);
                continue;
            }

            let sheet_gidx =
                doc.add_group(&format!("sheet: {}", sheet_name), GroupLabel::Section, None);
            let sheet_parent = format!("#/groups/{}", sheet_gidx);

            let merges = all_merges
                .get(sheet_name.as_str())
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            // Absolute start position of the range within the worksheet.
            // Merge regions use absolute coordinates, so we need this offset
            // to convert between grid-relative and absolute spaces.
            let (row_off, col_off) = range
                .start()
                .map(|(r, c)| (r as usize, c as usize))
                .unwrap_or((0, 0));

            let (eff_h, eff_w) = effective_dimensions(height, width, merges, row_off, col_off);

            if eff_h.saturating_mul(eff_w) > MAX_GRID_CELLS {
                log::warn!(
                    "Sheet '{}' grid {}x{} exceeds safety limit, skipping table detection",
                    sheet_name,
                    eff_h,
                    eff_w
                );
                continue;
            }

            let occupied = build_occupancy_grid(&range, eff_h, eff_w, merges, row_off, col_off);
            let tables = find_tables_flood_fill(&occupied, eff_h, eff_w);

            for bounds in &tables {
                let cells = extract_table_cells(&range, bounds, merges, row_off, col_off);
                let num_rows = bounds.max_row - bounds.min_row + 1;
                let num_cols = bounds.max_col - bounds.min_col + 1;
                if !cells.is_empty() {
                    doc.add_table(cells, num_rows as u32, num_cols as u32, Some(&sheet_parent));
                }
            }

            let n_images = images_per_sheet.get(sheet_name).copied().unwrap_or(0);
            for _ in 0..n_images {
                doc.add_picture(None, Some(&sheet_parent));
            }
        }

        Ok(doc)
    }
}

fn load_all_merged_regions(
    workbook: &mut calamine::Sheets<std::io::BufReader<std::fs::File>>,
    sheet_names: &[String],
) -> HashMap<String, Vec<Dimensions>> {
    let mut map = HashMap::new();
    if let calamine::Sheets::Xlsx(xlsx) = workbook {
        for name in sheet_names {
            if let Some(Ok(mut dims)) = xlsx.worksheet_merge_cells(name) {
                dims.retain(|d| d.start.0 <= d.end.0 && d.start.1 <= d.end.1);
                if !dims.is_empty() {
                    map.insert(name.clone(), dims);
                }
            }
        }
    }
    map
}

#[derive(Debug, Clone)]
struct TableBounds {
    min_row: usize,
    min_col: usize,
    max_row: usize,
    max_col: usize,
}

/// Expand grid dimensions to include merged regions that extend beyond the
/// data range.  Merge coordinates are absolute; we convert to grid-relative
/// using the supplied offsets.
fn effective_dimensions(
    height: usize,
    width: usize,
    merges: &[Dimensions],
    row_off: usize,
    col_off: usize,
) -> (usize, usize) {
    let mut h = height;
    let mut w = width;
    for dim in merges {
        let end_r = (dim.end.0 as usize).saturating_sub(row_off);
        let end_c = (dim.end.1 as usize).saturating_sub(col_off);
        h = h.max(end_r + 1);
        w = w.max(end_c + 1);
    }
    (h, w)
}

/// Build a boolean grid marking which cells are occupied (non-empty or part
/// of a merge region).  All coordinates inside the grid are 0-based relative
/// to the range start; merge coordinates are converted from absolute via
/// `row_off` / `col_off`.
fn build_occupancy_grid(
    range: &calamine::Range<Data>,
    height: usize,
    width: usize,
    merges: &[Dimensions],
    row_off: usize,
    col_off: usize,
) -> Vec<Vec<bool>> {
    let mut grid = vec![vec![false; width]; height];
    for (row_idx, row) in range.rows().enumerate() {
        if row_idx >= height {
            break;
        }
        for (col_idx, cell) in row.iter().enumerate() {
            if col_idx >= width {
                break;
            }
            let is_occupied = match cell {
                Data::Empty => false,
                Data::String(s) if s.trim().is_empty() => false,
                _ => true,
            };
            if is_occupied {
                grid[row_idx][col_idx] = true;
            }
        }
    }
    for dim in merges {
        let r_start = (dim.start.0 as usize).saturating_sub(row_off);
        let c_start = (dim.start.1 as usize).saturating_sub(col_off);
        let r_end = (dim.end.0 as usize).saturating_sub(row_off);
        let c_end = (dim.end.1 as usize).saturating_sub(col_off);
        for row in grid
            .iter_mut()
            .take(r_end.min(height.saturating_sub(1)) + 1)
            .skip(r_start)
        {
            for cell in row
                .iter_mut()
                .take(c_end.min(width.saturating_sub(1)) + 1)
                .skip(c_start)
            {
                *cell = true;
            }
        }
    }
    grid
}

/// BFS flood-fill to find connected regions of non-empty cells.
/// Uses 4-directional connectivity (no diagonals) matching Python's algorithm.
const GAP_TOLERANCE: usize = 0;

const DIRECTIONS: [(isize, isize); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

fn find_tables_flood_fill(occupied: &[Vec<bool>], height: usize, width: usize) -> Vec<TableBounds> {
    let mut visited = vec![vec![false; width]; height];
    let mut tables = Vec::new();

    for r in 0..height {
        for c in 0..width {
            if occupied[r][c] && !visited[r][c] {
                let bounds = bfs_region(occupied, &mut visited, r, c, height, width);
                tables.push(bounds);
            }
        }
    }

    tables
}

fn bfs_region(
    occupied: &[Vec<bool>],
    visited: &mut [Vec<bool>],
    start_r: usize,
    start_c: usize,
    height: usize,
    width: usize,
) -> TableBounds {
    let mut queue = VecDeque::new();
    queue.push_back((start_r, start_c));
    visited[start_r][start_c] = true;

    let mut min_row = start_r;
    let mut max_row = start_r;
    let mut min_col = start_c;
    let mut max_col = start_c;

    while let Some((r, c)) = queue.pop_front() {
        min_row = min_row.min(r);
        max_row = max_row.max(r);
        min_col = min_col.min(c);
        max_col = max_col.max(c);

        for &(dr, dc) in &DIRECTIONS {
            for step in 1..=(GAP_TOLERANCE as isize + 1) {
                let nr = r as isize + dr * step;
                let nc = c as isize + dc * step;
                if nr < 0 || nc < 0 || nr >= height as isize || nc >= width as isize {
                    break;
                }
                let (nr, nc) = (nr as usize, nc as usize);
                if visited[nr][nc] {
                    break;
                }
                if occupied[nr][nc] {
                    visited[nr][nc] = true;
                    queue.push_back((nr, nc));
                    break;
                }
            }
        }
    }

    TableBounds {
        min_row,
        min_col,
        max_row,
        max_col,
    }
}

/// Find the merge region whose top-left is exactly (row, col) in absolute
/// sheet coordinates.
fn find_merge_at(merges: &[Dimensions], row: usize, col: usize) -> Option<&Dimensions> {
    merges
        .iter()
        .find(|d| d.start.0 as usize == row && d.start.1 as usize == col)
}

/// Check whether (row, col) — in absolute sheet coordinates — falls inside a
/// merge region but is NOT the top-left anchor cell.
fn is_merge_continuation(merges: &[Dimensions], row: usize, col: usize) -> bool {
    merges.iter().any(|d| {
        let (r0, c0) = (d.start.0 as usize, d.start.1 as usize);
        let (r1, c1) = (d.end.0 as usize, d.end.1 as usize);
        row >= r0 && row <= r1 && col >= c0 && col <= c1 && !(row == r0 && col == c0)
    })
}

/// Extract `TableCell` items for one detected table region.  Grid-relative
/// bounds are converted to absolute coordinates for merge lookups, and spans
/// are clamped to the actual table extent.
fn extract_table_cells(
    range: &calamine::Range<Data>,
    bounds: &TableBounds,
    merges: &[Dimensions],
    row_off: usize,
    col_off: usize,
) -> Vec<TableCell> {
    let mut cells = Vec::new();
    let base_row = bounds.min_row;
    let base_col = bounds.min_col;

    for row_idx in bounds.min_row..=bounds.max_row {
        for col_idx in bounds.min_col..=bounds.max_col {
            let abs_row = row_idx + row_off;
            let abs_col = col_idx + col_off;

            if is_merge_continuation(merges, abs_row, abs_col) {
                continue;
            }

            let cell = range.get((row_idx, col_idx)).unwrap_or(&Data::Empty);
            let text = cell_to_string(cell);
            let rel_row = (row_idx - base_row) as u32;
            let rel_col = (col_idx - base_col) as u32;

            let (row_span, col_span) = if let Some(dim) = find_merge_at(merges, abs_row, abs_col) {
                let abs_table_max_row = bounds.max_row + row_off;
                let abs_table_max_col = bounds.max_col + col_off;
                let clamped_end_row = (dim.end.0 as usize).min(abs_table_max_row);
                let clamped_end_col = (dim.end.1 as usize).min(abs_table_max_col);
                let rs = clamped_end_row.saturating_sub(dim.start.0 as usize) + 1;
                let cs = clamped_end_col.saturating_sub(dim.start.1 as usize) + 1;
                ((rs as u32).max(1), (cs as u32).max(1))
            } else {
                (1, 1)
            };

            cells.push(TableCell {
                row_span,
                col_span,
                start_row_offset_idx: rel_row,
                end_row_offset_idx: rel_row + row_span,
                start_col_offset_idx: rel_col,
                end_col_offset_idx: rel_col + col_span,
                text,
                column_header: rel_row == 0,
                row_header: false,
                row_section: false,
                fillable: false,
                formatted_text: None,
            });
        }
    }

    cells
}

/// Count images per sheet by parsing the XLSX ZIP structure.
/// Returns a map from sheet name to number of images.
fn count_images_per_sheet(path: &Path, sheet_names: &[String]) -> HashMap<String, usize> {
    let mut result = HashMap::new();

    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return result,
    };
    let mut archive = match zip::ZipArchive::new(std::io::BufReader::new(file)) {
        Ok(a) => a,
        Err(_) => return result,
    };

    let wb_rels = parse_rels_from_zip(&mut archive, "xl/_rels/workbook.xml.rels");
    let sheet_file_map = parse_workbook_sheet_map(&mut archive, &wb_rels, sheet_names);

    for (sheet_name, sheet_file) in &sheet_file_map {
        let normalized = sheet_file
            .strip_prefix("/xl/")
            .or_else(|| sheet_file.strip_prefix("xl/"))
            .unwrap_or(sheet_file);
        let file_name = normalized.strip_prefix("worksheets/").unwrap_or(normalized);
        let rels_path = format!("xl/worksheets/_rels/{}.rels", file_name);
        let sheet_rels = parse_rels_from_zip(&mut archive, &rels_path);

        for (rel_type, target) in sheet_rels.values() {
            if !rel_type.contains("/drawing") {
                continue;
            }
            let drawing_xml_path = normalize_zip_path("xl/worksheets", target);

            let n = count_pic_elements_in_drawing(&mut archive, &drawing_xml_path);
            if n > 0 {
                *result.entry(sheet_name.clone()).or_insert(0) += n;
            }
        }
    }

    result
}

/// Resolve a relative or absolute target path within an XLSX ZIP.
fn normalize_zip_path(base_dir: &str, target: &str) -> String {
    if target.starts_with('/') {
        return target.trim_start_matches('/').to_string();
    }
    let mut parts: Vec<&str> = base_dir.split('/').collect();
    for seg in target.split('/') {
        match seg {
            ".." => {
                parts.pop();
            }
            "." | "" => {}
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// Parse a .rels XML file from the ZIP, returning rId -> (Type, Target).
fn parse_rels_from_zip(
    archive: &mut zip::ZipArchive<std::io::BufReader<std::fs::File>>,
    rels_path: &str,
) -> HashMap<String, (String, String)> {
    let mut map = HashMap::new();
    let mut xml_buf = String::new();

    if let Ok(mut f) = archive.by_name(rels_path) {
        let _ = f.read_to_string(&mut xml_buf);
    } else {
        return map;
    }

    let mut reader = quick_xml::Reader::from_str(&xml_buf);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Empty(ref e)) => {
                let ln = e.local_name();
                let local = String::from_utf8_lossy(ln.as_ref());
                if local == "Relationship" {
                    let mut id = String::new();
                    let mut rel_type = String::new();
                    let mut target = String::new();
                    for attr in e.attributes().flatten() {
                        let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                        let val = attr.unescape_value().unwrap_or_default().to_string();
                        match key.as_str() {
                            "Id" => id = val,
                            "Type" => rel_type = val,
                            "Target" => target = val,
                            _ => {}
                        }
                    }
                    if !id.is_empty() {
                        map.insert(id, (rel_type, target));
                    }
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    map
}

/// Parse xl/workbook.xml to map sheet names to their file targets.
fn parse_workbook_sheet_map(
    archive: &mut zip::ZipArchive<std::io::BufReader<std::fs::File>>,
    wb_rels: &HashMap<String, (String, String)>,
    sheet_names: &[String],
) -> Vec<(String, String)> {
    let mut xml_buf = String::new();
    if let Ok(mut f) = archive.by_name("xl/workbook.xml") {
        let _ = f.read_to_string(&mut xml_buf);
    } else {
        return Vec::new();
    }

    let mut ordered: Vec<(String, String)> = Vec::new();

    let mut reader = quick_xml::Reader::from_str(&xml_buf);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Start(ref e))
            | Ok(quick_xml::events::Event::Empty(ref e)) => {
                let ln = e.local_name();
                let local = String::from_utf8_lossy(ln.as_ref());
                if local == "sheet" {
                    let mut name = String::new();
                    let mut rid = String::new();
                    for attr in e.attributes().flatten() {
                        let key = String::from_utf8_lossy(attr.key.as_ref()).to_string();
                        let val = attr.unescape_value().unwrap_or_default().to_string();
                        if key == "name" {
                            name = val;
                        } else if key.ends_with("id") {
                            rid = val;
                        }
                    }
                    if !name.is_empty() && !rid.is_empty() {
                        if let Some((_, target)) = wb_rels.get(&rid) {
                            ordered.push((name, target.clone()));
                        }
                    }
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    sheet_names
        .iter()
        .filter_map(|sn| {
            ordered
                .iter()
                .find(|(name, _)| name == sn)
                .map(|(_, target)| (sn.clone(), target.clone()))
        })
        .collect()
}

/// Count <pic> elements inside a drawing XML file.
fn count_pic_elements_in_drawing(
    archive: &mut zip::ZipArchive<std::io::BufReader<std::fs::File>>,
    drawing_path: &str,
) -> usize {
    let mut xml_buf = String::new();
    if let Ok(mut f) = archive.by_name(drawing_path) {
        let _ = f.read_to_string(&mut xml_buf);
    } else {
        return 0;
    }

    let mut count = 0;
    let mut reader = quick_xml::Reader::from_str(&xml_buf);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(quick_xml::events::Event::Start(ref e)) => {
                let ln = e.local_name();
                let local = String::from_utf8_lossy(ln.as_ref());
                if local == "pic" {
                    count += 1;
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    count
}

fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::Int(i) => i.to_string(),
        Data::Float(f) => {
            if f.is_nan() || f.is_infinite() {
                return String::new();
            }
            if *f == f.floor() && f.abs() < 1e15 {
                format!("{}", *f as i64)
            } else {
                normalize_float(*f)
            }
        }
        Data::String(s) => s.clone(),
        Data::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Data::DateTime(dt) => match dt.as_datetime() {
            Some(naive) => format!("{}", naive.format("%Y-%m-%d %H:%M:%S")),
            None => format!("{}", dt),
        },
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => match e {
            calamine::CellErrorType::Div0 => "#DIV/0!".to_string(),
            calamine::CellErrorType::NA => "#N/A".to_string(),
            calamine::CellErrorType::Name => "#NAME?".to_string(),
            calamine::CellErrorType::Null => "#NULL!".to_string(),
            calamine::CellErrorType::Num => "#NUM!".to_string(),
            calamine::CellErrorType::Ref => "#REF!".to_string(),
            calamine::CellErrorType::Value => "#VALUE!".to_string(),
            calamine::CellErrorType::GettingData => "#GETTING_DATA".to_string(),
        },
        Data::Empty => String::new(),
    }
}

fn normalize_float(f: f64) -> String {
    for precision in 0..=15 {
        let rounded = format!("{:.prec$}", f, prec = precision);
        if let Ok(parsed) = rounded.parse::<f64>() {
            if (parsed - f).abs() < f64::EPSILON * f.abs().max(1.0) * 10.0 {
                if rounded.contains('.') {
                    return rounded.trim_end_matches('0').trim_end_matches('.').to_string();
                }
                return rounded;
            }
        }
    }
    f.to_string()
}
