use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Context;
use base64::Engine;
use lopdf::content::Operation;
use lopdf::{Document, Encoding, Object, ObjectId};

use crate::models::common::{DocItemLabel, GroupLabel, InputFormat};
use crate::models::document::{compute_hash, doc_name_from_path, DoclingDocument};
use crate::models::page::{BoundingBox, PageItem, ProvenanceItem, Size};
use crate::models::picture::{ImageRef, ImageSize};
use crate::models::table::TableCell;

use super::Backend;

// Pdfium for page rendering
use pdfium_render::prelude::*;

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Bullet glyphs that PDF documents use for unordered list items.
/// These are the only characters treated as "stray markers" during
/// paragraph fragment merging -- they have no other valid meaning when
/// they appear as a standalone character on a line.
const BULLET_GLYPHS: &[char] = &['•', '○', '■', '□', '◦', '▪'];

// ---------------------------------------------------------------------------
// Layout analysis structures (from content stream, no text decoding)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TextEvent {
    x: f64,
    _y: f64,
    font_size: f64,
    _byte_len: usize,
}

#[derive(Debug, Clone)]
struct PositionedText {
    text: String,
    x: f64,
    y: f64,
    _font_size: f64,
}

#[derive(Debug, Clone)]
struct PathSegment {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

impl PathSegment {
    fn is_horizontal(&self, tol: f64) -> bool {
        (self.y1 - self.y2).abs() < tol
    }
    fn is_vertical(&self, tol: f64) -> bool {
        (self.x1 - self.x2).abs() < tol
    }
}

#[derive(Debug, Clone)]
struct ImageInfo {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    data: Option<Vec<u8>>,
    mimetype: Option<String>,
}

struct LayoutInfo {
    text_events: Vec<TextEvent>,
    positioned_texts: Vec<PositionedText>,
    paths: Vec<PathSegment>,
    images: Vec<ImageInfo>,
}

#[derive(Clone)]
struct GState {
    ctm: [f64; 6],
}

impl Default for GState {
    fn default() -> Self {
        GState {
            ctm: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }
}

fn ctm_multiply(a: &[f64; 6], b: &[f64; 6]) -> [f64; 6] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}

fn ctm_transform(ctm: &[f64; 6], x: f64, y: f64) -> (f64, f64) {
    (
        ctm[0] * x + ctm[2] * y + ctm[4],
        ctm[1] * x + ctm[3] * y + ctm[5],
    )
}

// ---------------------------------------------------------------------------
// Content stream analysis
// ---------------------------------------------------------------------------

fn analyze_page_layout(
    doc: &Document,
    page_id: ObjectId,
) -> anyhow::Result<LayoutInfo> {
    let fonts = doc.get_page_fonts(page_id).unwrap_or_default();
    let encodings: std::collections::BTreeMap<Vec<u8>, Encoding> = fonts
        .into_iter()
        .filter_map(|(name, font)| font.get_font_encoding(doc).ok().map(|enc| (name, enc)))
        .collect();

    let content = doc.get_and_decode_page_content(page_id)?;

    let mut text_events: Vec<TextEvent> = Vec::new();
    let mut positioned_texts: Vec<PositionedText> = Vec::new();
    let mut paths: Vec<PathSegment> = Vec::new();
    let mut images: Vec<ImageInfo> = Vec::new();

    let mut gs_stack: Vec<GState> = vec![GState::default()];
    let mut current_encoding: Option<&Encoding> = None;
    let mut font_size: f64 = 12.0;
    let mut tm_x: f64 = 0.0;
    let mut tm_y: f64 = 0.0;
    let mut line_x: f64 = 0.0;
    let mut line_y: f64 = 0.0;
    let mut leading: f64 = 0.0;
    let mut tm_scale_y: f64 = 1.0;

    let mut path_x: f64 = 0.0;
    let mut path_y: f64 = 0.0;
    let mut pending_segments: Vec<PathSegment> = Vec::new();
    let mut subpath_start: (f64, f64) = (0.0, 0.0);

    let xobjects = get_page_xobjects(doc, page_id);

    for Operation { operator, operands } in &content.operations {
        let gs = gs_stack.last().cloned().unwrap_or_default();

        match operator.as_str() {
            "q" => gs_stack.push(gs.clone()),
            "Q" => {
                if gs_stack.len() > 1 {
                    gs_stack.pop();
                }
            }
            "cm" => {
                if operands.len() >= 6 {
                    let m = [
                        obj_as_f64(&operands[0]).unwrap_or(1.0),
                        obj_as_f64(&operands[1]).unwrap_or(0.0),
                        obj_as_f64(&operands[2]).unwrap_or(0.0),
                        obj_as_f64(&operands[3]).unwrap_or(1.0),
                        obj_as_f64(&operands[4]).unwrap_or(0.0),
                        obj_as_f64(&operands[5]).unwrap_or(0.0),
                    ];
                    if let Some(top) = gs_stack.last_mut() {
                        top.ctm = ctm_multiply(&m, &top.ctm);
                    }
                }
            }

            "BT" => {
                tm_x = 0.0;
                tm_y = 0.0;
                line_x = 0.0;
                line_y = 0.0;
                tm_scale_y = 1.0;
            }
            "ET" => {}
            "Tf" => {
                if let Some(name) = operands.first().and_then(|o| o.as_name().ok()) {
                    current_encoding = encodings.get(name);
                }
                if operands.len() >= 2 {
                    if let Some(s) = obj_as_f64(&operands[1]) {
                        if s.abs() > 0.1 {
                            font_size = s.abs();
                        }
                    }
                }
            }
            "TL" => {
                if let Some(tl) = operands.first().and_then(|o| obj_as_f64(o)) {
                    leading = tl;
                }
            }
            "Tm" => {
                if operands.len() >= 6 {
                    let d = obj_as_f64(&operands[3]).unwrap_or(1.0);
                    let e = obj_as_f64(&operands[4]).unwrap_or(0.0);
                    let f = obj_as_f64(&operands[5]).unwrap_or(0.0);
                    tm_x = e;
                    tm_y = f;
                    line_x = e;
                    line_y = f;
                    tm_scale_y = d.abs().max(0.1);
                }
            }
            "Td" => {
                if operands.len() >= 2 {
                    let tx = obj_as_f64(&operands[0]).unwrap_or(0.0);
                    let ty = obj_as_f64(&operands[1]).unwrap_or(0.0);
                    line_x += tx;
                    line_y += ty;
                    tm_x = line_x;
                    tm_y = line_y;
                }
            }
            "TD" => {
                if operands.len() >= 2 {
                    let tx = obj_as_f64(&operands[0]).unwrap_or(0.0);
                    let ty = obj_as_f64(&operands[1]).unwrap_or(0.0);
                    line_x += tx;
                    line_y += ty;
                    tm_x = line_x;
                    tm_y = line_y;
                    leading = -ty;
                }
            }
            "T*" => {
                line_y -= leading;
                tm_x = line_x;
                tm_y = line_y;
            }
            "Tj" | "TJ" | "'" | "\"" => {
                let effective_size = font_size * tm_scale_y;
                let (gx, gy) = ctm_transform(&gs.ctm, tm_x, tm_y);
                let byte_len = operand_byte_len(operands);

                text_events.push(TextEvent {
                    x: gx,
                    _y: gy,
                    font_size: effective_size,
                    _byte_len: byte_len,
                });

                if let Some(encoding) = current_encoding {
                    let ops = match operator.as_str() {
                        "\"" if operands.len() >= 3 => &operands[2..],
                        _ => operands.as_slice(),
                    };
                    let text = decode_operands(encoding, ops);
                    if !text.is_empty() {
                        positioned_texts.push(PositionedText {
                            text,
                            x: gx,
                            y: gy,
                            _font_size: effective_size,
                        });
                    }
                }

                if operator == "'" || operator == "\"" {
                    line_y -= leading;
                    tm_x = line_x;
                    tm_y = line_y;
                }
            }

            // Path construction
            "m" => {
                if operands.len() >= 2 {
                    let x = obj_as_f64(&operands[0]).unwrap_or(0.0);
                    let y = obj_as_f64(&operands[1]).unwrap_or(0.0);
                    let (gx, gy) = ctm_transform(&gs.ctm, x, y);
                    path_x = gx;
                    path_y = gy;
                    subpath_start = (gx, gy);
                }
            }
            "l" => {
                if operands.len() >= 2 {
                    let x = obj_as_f64(&operands[0]).unwrap_or(0.0);
                    let y = obj_as_f64(&operands[1]).unwrap_or(0.0);
                    let (gx, gy) = ctm_transform(&gs.ctm, x, y);
                    pending_segments.push(PathSegment {
                        x1: path_x,
                        y1: path_y,
                        x2: gx,
                        y2: gy,
                    });
                    path_x = gx;
                    path_y = gy;
                }
            }
            "re" => {
                if operands.len() >= 4 {
                    let rx = obj_as_f64(&operands[0]).unwrap_or(0.0);
                    let ry = obj_as_f64(&operands[1]).unwrap_or(0.0);
                    let rw = obj_as_f64(&operands[2]).unwrap_or(0.0);
                    let rh = obj_as_f64(&operands[3]).unwrap_or(0.0);
                    let (x0, y0) = ctm_transform(&gs.ctm, rx, ry);
                    let (x1, y1) = ctm_transform(&gs.ctm, rx + rw, ry + rh);
                    pending_segments.push(PathSegment { x1: x0, y1: y0, x2: x1, y2: y0 });
                    pending_segments.push(PathSegment { x1: x1, y1: y0, x2: x1, y2: y1 });
                    pending_segments.push(PathSegment { x1: x1, y1: y1, x2: x0, y2: y1 });
                    pending_segments.push(PathSegment { x1: x0, y1: y1, x2: x0, y2: y0 });
                }
            }
            "h" => {
                if (path_x - subpath_start.0).abs() > 0.01
                    || (path_y - subpath_start.1).abs() > 0.01
                {
                    pending_segments.push(PathSegment {
                        x1: path_x,
                        y1: path_y,
                        x2: subpath_start.0,
                        y2: subpath_start.1,
                    });
                }
                path_x = subpath_start.0;
                path_y = subpath_start.1;
            }
            "S" | "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" => {
                paths.extend(pending_segments.drain(..));
            }
            "n" => {
                pending_segments.clear();
            }

            "Do" => {
                if let Some(name) = operands.first().and_then(|o| o.as_name().ok()) {
                    if let Some(xobj_id) = xobjects.get(name) {
                        if let Ok(obj) = doc.get_object(*xobj_id) {
                            if let Ok(stream) = obj.as_stream() {
                                let dict = &stream.dict;
                                let subtype = dict
                                    .get(b"Subtype")
                                    .ok()
                                    .and_then(|o| o.as_name().ok())
                                    .unwrap_or(b"");
                                if subtype == b"Image" {
                                    let (gx, gy) = ctm_transform(&gs.ctm, 0.0, 0.0);
                                    let w = (gs.ctm[0].powi(2) + gs.ctm[1].powi(2)).sqrt();
                                    let h = (gs.ctm[2].powi(2) + gs.ctm[3].powi(2)).sqrt();

                                    let (data, mimetype) =
                                        extract_image_bytes(doc, stream, *xobj_id);

                                    images.push(ImageInfo {
                                        x: gx,
                                        y: gy,
                                        width: w,
                                        height: h,
                                        data,
                                        mimetype,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(LayoutInfo {
        text_events,
        positioned_texts,
        paths,
        images,
    })
}

fn operand_byte_len(operands: &[Object]) -> usize {
    let mut len = 0;
    for op in operands {
        match op {
            Object::String(bytes, _) => len += bytes.len(),
            Object::Array(arr) => {
                for item in arr {
                    if let Object::String(bytes, _) = item {
                        len += bytes.len();
                    }
                }
            }
            _ => {}
        }
    }
    len
}

fn get_page_xobjects(doc: &Document, page_id: ObjectId) -> HashMap<Vec<u8>, ObjectId> {
    let mut result = HashMap::new();
    let page_dict = match doc.get_object(page_id).ok().and_then(|o| o.as_dict().ok()) {
        Some(d) => d,
        None => return result,
    };

    let resources = match page_dict.get(b"Resources") {
        Ok(obj) => resolve_dict(doc, obj),
        Err(_) => {
            if let Ok(parent_ref) = page_dict.get(b"Parent") {
                if let Ok(parent_id) = parent_ref.as_reference() {
                    if let Ok(parent_obj) = doc.get_object(parent_id) {
                        if let Ok(parent_dict) = parent_obj.as_dict() {
                            parent_dict.get(b"Resources").ok().and_then(|o| resolve_dict(doc, o))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
    };

    if let Some(res_dict) = resources {
        if let Ok(xobj_obj) = res_dict.get(b"XObject") {
            if let Some(xobj_dict) = resolve_dict(doc, xobj_obj) {
                for (name, val) in xobj_dict.iter() {
                    if let Ok(id) = val.as_reference() {
                        result.insert(name.clone(), id);
                    }
                }
            }
        }
    }

    result
}

fn resolve_dict<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a lopdf::Dictionary> {
    match obj {
        Object::Dictionary(d) => Some(d),
        Object::Reference(r) => doc.get_object(*r).ok().and_then(|o| o.as_dict().ok()),
        _ => None,
    }
}

fn decode_operands(encoding: &Encoding, operands: &[Object]) -> String {
    let mut text = String::new();
    for operand in operands {
        match operand {
            Object::String(bytes, _) => {
                if let Ok(decoded) = Document::decode_text(encoding, bytes) {
                    text.push_str(&decoded);
                }
            }
            Object::Array(arr) => {
                for item in arr {
                    match item {
                        Object::String(bytes, _) => {
                            if let Ok(decoded) = Document::decode_text(encoding, bytes) {
                                text.push_str(&decoded);
                            }
                        }
                        Object::Integer(i) if *i < -100 => text.push(' '),
                        Object::Real(f) if *f < -100.0 => text.push(' '),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    text
}

// ---------------------------------------------------------------------------
// Column detection from layout info
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum ColumnLayout {
    Single,
    TwoColumn { boundary: f64 },
}

fn detect_columns(events: &[TextEvent], page_width: f64) -> ColumnLayout {
    if events.len() < 10 {
        return ColumnLayout::Single;
    }

    let midpoint = page_width / 2.0;
    let margin = page_width * 0.08;

    let mut left_count = 0;
    let mut right_count = 0;

    for ev in events {
        if ev.x > margin && ev.x < midpoint - margin {
            left_count += 1;
        }
        if ev.x > midpoint + margin && ev.x < page_width - margin {
            right_count += 1;
        }
    }

    if left_count >= 5 && right_count >= 5 {
        ColumnLayout::TwoColumn { boundary: midpoint }
    } else {
        ColumnLayout::Single
    }
}

// ---------------------------------------------------------------------------
// Font size analysis for classification
// ---------------------------------------------------------------------------

struct FontSizeInfo {
    _body_size: f64,
    sizes_per_event: Vec<f64>,
}

fn analyze_font_sizes(events: &[TextEvent]) -> FontSizeInfo {
    let sizes: Vec<f64> = events.iter().map(|e| e.font_size).collect();
    let body_size = median_f64(&sizes);
    FontSizeInfo {
        _body_size: body_size,
        sizes_per_event: sizes,
    }
}

fn median_f64(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 12.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

// ---------------------------------------------------------------------------
// Table detection from ruled lines
// ---------------------------------------------------------------------------

struct TableRegion {
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
    h_lines: Vec<f64>,
    v_lines: Vec<f64>,
}

fn detect_table_regions(paths: &[PathSegment], page_height: f64) -> Vec<TableRegion> {
    let tol = 2.0;

    let mut h_lines: Vec<(f64, f64, f64)> = Vec::new();
    let mut v_lines: Vec<(f64, f64, f64)> = Vec::new();

    for seg in paths {
        let y1 = page_height - seg.y1;
        let y2 = page_height - seg.y2;

        if seg.is_horizontal(tol) {
            let x_start = seg.x1.min(seg.x2);
            let x_end = seg.x1.max(seg.x2);
            if (x_end - x_start) > 20.0 {
                h_lines.push((y1, x_start, x_end));
            }
        }
        if seg.is_vertical(tol) {
            let y_start = y1.min(y2);
            let y_end = y1.max(y2);
            if (y_end - y_start) > 10.0 {
                v_lines.push((seg.x1, y_start, y_end));
            }
        }
    }

    // Require enough lines for a real table grid (at least 3 rows, 2 cols)
    if h_lines.len() < 4 || v_lines.len() < 3 {
        return Vec::new();
    }

    h_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    v_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let x_range = h_lines.iter().fold(
        (f64::MAX, f64::MIN),
        |(min, max), (_, xs, xe)| (min.min(*xs), max.max(*xe)),
    );
    let y_range = (
        h_lines.first().map(|l| l.0).unwrap_or(0.0),
        h_lines.last().map(|l| l.0).unwrap_or(0.0),
    );

    let v_in_range: Vec<f64> = v_lines
        .iter()
        .filter(|(x, ys, ye)| {
            *x >= x_range.0 - tol
                && *x <= x_range.1 + tol
                && *ys <= y_range.1 + tol
                && *ye >= y_range.0 - tol
        })
        .map(|(x, _, _)| *x)
        .collect();

    if v_in_range.len() >= 3 {
        let unique_y = dedup_sorted(&h_lines.iter().map(|l| l.0).collect::<Vec<_>>(), tol);
        let unique_x = dedup_sorted(&v_in_range, tol);

        if unique_y.len() >= 3 && unique_x.len() >= 3 {
            return vec![TableRegion {
                x_min: x_range.0,
                y_min: y_range.0,
                x_max: x_range.1,
                y_max: y_range.1,
                h_lines: unique_y,
                v_lines: unique_x,
            }];
        }
    }

    Vec::new()
}

fn dedup_sorted(vals: &[f64], tol: f64) -> Vec<f64> {
    let mut result = Vec::new();
    for &v in vals {
        if result
            .last()
            .map_or(true, |&last: &f64| (v - last).abs() > tol)
        {
            result.push(v);
        }
    }
    result
}

// Assign positioned text fragments to table cells
fn build_table_from_texts(
    texts: &[PositionedText],
    region: &TableRegion,
    page_height: f64,
) -> Option<(Vec<Vec<String>>, usize, usize)> {
    let num_rows = region.h_lines.len().saturating_sub(1);
    let num_cols = region.v_lines.len().saturating_sub(1);
    if num_rows == 0 || num_cols == 0 {
        return None;
    }

    let mut grid: Vec<Vec<String>> = vec![vec![String::new(); num_cols]; num_rows];

    for pt in texts {
        let ty = page_height - pt.y;
        if ty < region.y_min - 5.0 || ty > region.y_max + 5.0 {
            continue;
        }
        if pt.x < region.x_min - 5.0 || pt.x > region.x_max + 5.0 {
            continue;
        }

        let col = region
            .v_lines
            .windows(2)
            .position(|w| pt.x >= w[0] - 5.0 && pt.x <= w[1] + 5.0)
            .unwrap_or(0)
            .min(num_cols - 1);
        let row = region
            .h_lines
            .windows(2)
            .position(|w| ty >= w[0] - 5.0 && ty <= w[1] + 5.0)
            .unwrap_or(0)
            .min(num_rows - 1);

        if !grid[row][col].is_empty() {
            grid[row][col].push(' ');
        }
        grid[row][col].push_str(pt.text.trim());
    }

    Some((grid, num_rows, num_cols))
}

// ---------------------------------------------------------------------------
// Text processing and classification
// ---------------------------------------------------------------------------

fn sanitize_text(text: &str) -> String {
    let mut result = text.to_string();

    // Ligatures
    result = result
        .replace('\u{FB00}', "ff")
        .replace('\u{FB01}', "fi")
        .replace('\u{FB02}', "fl")
        .replace('\u{FB03}', "ffi")
        .replace('\u{FB04}', "ffl")
        .replace('\u{FB05}', "st")
        .replace('\u{FB06}', "st");

    // Unicode normalization (matching Python docling behaviour)
    result = result
        .replace('\u{2044}', "/")  // fraction slash
        .replace('\u{2019}', "'")  // right single quote
        .replace('\u{2018}', "'")  // left single quote
        .replace('\u{201C}', "\"") // left double quote
        .replace('\u{201D}', "\"") // right double quote
        .replace('\u{00A0}', " "); // non-breaking space

    // Hyphenation joining and whitespace collapsing
    let mut out = String::with_capacity(result.len());
    let chars: Vec<char> = result.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Join hyphenated words split across lines
        if chars[i] == '-'
            && i + 1 < chars.len()
            && chars[i + 1] == '\n'
            && i > 0
            && chars[i - 1].is_alphabetic()
        {
            let next_alpha = chars[i + 2..].iter().find(|c| !c.is_whitespace());
            if next_alpha.is_some_and(|c| c.is_lowercase()) {
                i += 2;
                while i < chars.len() && chars[i].is_whitespace() && chars[i] != '\n' {
                    i += 1;
                }
                continue;
            }
        }

        // Collapse runs of spaces (not newlines) into a single space
        if chars[i] == ' ' && i + 1 < chars.len() && chars[i + 1] == ' ' {
            out.push(' ');
            while i < chars.len() && chars[i] == ' ' {
                i += 1;
            }
            continue;
        }

        out.push(chars[i]);
        i += 1;
    }

    out
}

fn split_paragraphs(text: &str) -> Vec<&str> {
    text.split("\n\n")
        .flat_map(|block| split_single_newline_paragraphs(block))
        .filter(|s| !s.trim().is_empty())
        .collect()
}

fn split_single_newline_paragraphs(block: &str) -> Vec<&str> {
    let lines: Vec<&str> = block.lines().collect();
    if lines.len() <= 1 {
        return vec![block];
    }

    let avg_len = {
        let total: usize = lines.iter().map(|l| l.trim().len()).sum();
        (total as f64 / lines.len() as f64).max(1.0)
    };
    
    // For presentation-style content (many short lines), split more aggressively
    let is_presentation_style = avg_len < 60.0 && lines.len() > 3;

    let mut result = Vec::new();
    let mut start = 0;
    let block_bytes = block.as_bytes();
    let mut byte_offset = 0;

    for (i, line) in lines.iter().enumerate() {
        let line_start = byte_offset;
        byte_offset += line.len();
        if i < lines.len() - 1 {
            byte_offset += 1;
        }
        if i < lines.len() - 1 {
            let trimmed = line.trim();
            let trimmed_len = trimmed.len();
            
            // Split condition: either short relative to average, or presentation-style
            // where we split on lines that look like headers/standalone text
            let should_split = if is_presentation_style {
                // In presentation style, split after short lines that don't end with
                // continuation markers (comma, hyphen at end of word)
                trimmed_len > 0 && trimmed_len < 80 
                    && !trimmed.ends_with(',')
                    && !trimmed.ends_with('-')
                    && !trimmed.ends_with(':')
            } else {
                trimmed_len > 0 && (trimmed_len as f64) < avg_len * 0.4
            };
            
            if should_split {
                let end = line_start + line.len();
                let segment = &block[start..end];
                if !segment.trim().is_empty() {
                    result.push(segment.trim());
                }
                start = end;
                while start < block_bytes.len() && block_bytes[start] == b'\n' {
                    start += 1;
                }
            }
        }
    }

    if start < block.len() {
        let tail = &block[start..];
        if !tail.trim().is_empty() {
            result.push(tail.trim());
        }
    }

    if result.is_empty() {
        vec![block]
    } else {
        result
    }
}

/// Reassociate stray bullet glyphs that PDF text extraction attached to the
/// wrong paragraph.
///
/// PDF layout often produces text like `"Item1 \n•"` where the trailing `•`
/// actually belongs to the *next* list item. This function strips those
/// trailing bullets and prepends them to the following paragraph, then drops
/// any paragraphs that were left empty or consist only of a lone bullet.
fn merge_fragment_paragraphs(paragraphs: &[&str]) -> Vec<String> {
    let mut result: Vec<String> = Vec::with_capacity(paragraphs.len());
    let mut carry_bullet: Option<char> = None;

    for &para in paragraphs {
        let trimmed = para.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Prepend any bullet carried from the previous paragraph
        let mut text = if let Some(bullet) = carry_bullet.take() {
            format!("{} {}", bullet, trimmed)
        } else {
            trimmed.to_string()
        };

        // Strip a leading `<bullet>\n` — the bullet was a leftover from the
        // previous text block, not a marker for this paragraph.
        if let Some(first_char) = text.chars().next() {
            if BULLET_GLYPHS.contains(&first_char) {
                let after = &text[first_char.len_utf8()..];
                if after.starts_with('\n') {
                    text = after[1..].to_string();
                }
            }
        }

        // Detect a trailing `\n<bullet>` and carry the bullet forward.
        if let Some(last_char) = text.trim_end().chars().next_back() {
            if BULLET_GLYPHS.contains(&last_char) {
                let trimmed = text.trim_end();
                let before_bullet = &trimmed[..trimmed.len() - last_char.len_utf8()];
                // The bullet must be preceded by a newline (possibly with spaces)
                if before_bullet.ends_with('\n')
                    || before_bullet.trim_end().ends_with('\n')
                    || before_bullet.trim().is_empty()
                {
                    carry_bullet = Some(last_char);
                    text = before_bullet.trim_end().trim_end_matches('\n').to_string();
                }
            }
        }

        let text = text.trim();
        if text.is_empty() {
            continue;
        }

        // Drop paragraphs that are a lone bullet glyph (after all stripping)
        if text.chars().count() == 1 && BULLET_GLYPHS.contains(&text.chars().next().unwrap()) {
            carry_bullet = Some(text.chars().next().unwrap());
            continue;
        }

        result.push(text.to_string());
    }

    // A trailing carried bullet with nothing after it is discarded.

    result
}

fn classify_paragraph(
    text: &str,
    body_font_size: f64,
    local_font_size: Option<f64>,
) -> DocItemLabel {
    let trimmed = text.trim();

    if looks_like_caption(trimmed) {
        return DocItemLabel::Caption;
    }

    // Check list items early so they aren't mis-classified as headings
    if looks_like_list_item(trimmed) {
        return DocItemLabel::ListItem;
    }

    // Structural section header patterns (numbered sections, "Chapter", etc.)
    if looks_like_section_header(trimmed) {
        return DocItemLabel::SectionHeader;
    }

    // Font-size based heading detection (strong signal)
    if let Some(fs) = local_font_size {
        let ratio = fs / body_font_size;
        if ratio >= 1.3 && trimmed.len() < 200 && trimmed.lines().count() <= 3 {
            return DocItemLabel::SectionHeader;
        }
    }

    // Short single-line text: only promote to heading with strong font-size evidence
    // Be conservative - require larger font difference to avoid promoting diagram labels
    let line_count = trimmed.lines().count();
    let char_count = trimmed.len();
    if line_count == 1 && char_count < 80 && char_count > 3 {
        if !trimmed.contains(". ")
            && !trimmed.ends_with('.')
            && !trimmed.ends_with(',')
            && !trimmed.ends_with(';')
            && !trimmed.ends_with(')')
        {
            if let Some(fs) = local_font_size {
                let ratio = fs / body_font_size;
                // Require stronger signal (1.25) for short lines without clear header patterns
                if ratio >= 1.25 {
                    return DocItemLabel::SectionHeader;
                }
            }
        }
    }

    if let Some(fs) = local_font_size {
        let ratio = fs / body_font_size;
        if ratio < 0.85 && trimmed.len() < 300 {
            if trimmed.starts_with(|c: char| c.is_ascii_digit())
                || trimmed.starts_with('*')
                || trimmed.starts_with('†')
            {
                return DocItemLabel::Footnote;
            }
        }
    }

    DocItemLabel::Text
}

fn starts_with_bullet(text: &str) -> bool {
    if let Some(first) = text.chars().next() {
        if BULLET_GLYPHS.contains(&first) {
            let rest = &text[first.len_utf8()..];
            return rest.starts_with(' ') || rest.is_empty();
        }
        // Hyphen-minus and typographic dashes followed by a space
        if (first == '-' || first == '–' || first == '—') && text.len() > 2 {
            return text[first.len_utf8()..].starts_with(' ');
        }
    }
    false
}

fn starts_with_ordered_marker(text: &str) -> Option<usize> {
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();

    // "1. ", "12) " style
    if !bytes.is_empty() && bytes[0].is_ascii_digit() {
        if let Some(pos) = trimmed.find(|c: char| !c.is_ascii_digit()) {
            let after = &trimmed[pos..];
            if after.starts_with(". ") || after.starts_with(") ") {
                return Some(pos + 1); // length up to and including the `.` or `)`
            }
        }
    }
    // "(a) " style
    if trimmed.starts_with('(') {
        if let Some(close) = trimmed.find(')') {
            if close < 6 && trimmed.len() > close + 2 && trimmed.as_bytes()[close + 1] == b' ' {
                return Some(close + 1);
            }
        }
    }
    None
}

fn looks_like_list_item(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.len() > 500 {
        return false;
    }
    starts_with_bullet(trimmed) || starts_with_ordered_marker(trimmed).is_some()
}

fn looks_like_numbered_list(text: &str) -> bool {
    starts_with_ordered_marker(text.trim()).is_some()
}

/// Return the marker string for a list item (e.g. `"•"`, `"-"`, `"1."`, `"(a)"`).
fn extract_list_marker(text: &str) -> Option<String> {
    let trimmed = text.trim();

    if let Some(first) = trimmed.chars().next() {
        if BULLET_GLYPHS.contains(&first) || first == '-' || first == '–' || first == '—' {
            let rest = &trimmed[first.len_utf8()..];
            if rest.starts_with(' ') || rest.is_empty() {
                return Some(first.to_string());
            }
        }
    }

    if let Some(marker_end) = starts_with_ordered_marker(trimmed) {
        return Some(trimmed[..marker_end].to_string());
    }

    None
}

/// Strip the leading marker (and trailing space) from list-item text.
fn strip_list_marker(text: &str) -> String {
    let trimmed = text.trim();

    // Bullet or dash marker: single char + space
    if let Some(first) = trimmed.chars().next() {
        if BULLET_GLYPHS.contains(&first) || first == '-' || first == '–' || first == '—' {
            let rest = &trimmed[first.len_utf8()..];
            if rest.starts_with(' ') {
                return rest[1..].trim_start().to_string();
            }
        }
    }

    // Ordered marker: "1. text" or "(a) text"
    if let Some(marker_end) = starts_with_ordered_marker(trimmed) {
        let after = &trimmed[marker_end..];
        return after.trim_start().to_string();
    }

    trimmed.to_string()
}

fn looks_like_caption(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.len() > 200 {
        return false;
    }
    let lower = trimmed.to_lowercase();
    lower.starts_with("figure ")
        || lower.starts_with("fig. ")
        || lower.starts_with("fig ")
        || lower.starts_with("table ")
        || lower.starts_with("tab. ")
        || lower.starts_with("listing ")
        || lower.starts_with("algorithm ")
        || lower.starts_with("scheme ")
}

/// Detect headings with explicit structural numbering.
///
/// Matches two reliable patterns that don't depend on font size:
///   1. Leading section number: `"1 Intro"`, `"1.2 Methods"`, `"A.1 Appendix"`
///   2. Structural keyword + identifier: `"Chapter 3"`, `"Part II"`,
///      `"Section 1.2"`, `"Appendix A"`
fn looks_like_section_header(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.len() > 120 || trimmed.lines().count() > 2 {
        return false;
    }

    let first_word = match trimmed.split_whitespace().next() {
        Some(w) => w,
        None => return false,
    };
    let cleaned = first_word.trim_end_matches('.');
    if cleaned.is_empty() {
        return false;
    }

    // Pattern 1: leading section number — "1", "1.2", "1.2.3"
    if cleaned.chars().all(|c| c.is_ascii_digit() || c == '.') && cleaned.chars().any(|c| c.is_ascii_digit()) {
        let rest = trimmed.splitn(2, char::is_whitespace).nth(1).unwrap_or("");
        return !rest.is_empty() && rest.len() < 100;
    }

    // Pattern 1b: letter-number — "A", "A.1", "B.2.3"
    let mut chars = cleaned.chars();
    if let Some(first_ch) = chars.next() {
        if first_ch.is_ascii_uppercase() {
            let rest_str: String = chars.collect();
            let is_section_number = rest_str.is_empty()
                || rest_str.chars().all(|c| c.is_ascii_digit() || c == '.');
            if is_section_number {
                let rest = trimmed.splitn(2, char::is_whitespace).nth(1).unwrap_or("");
                return !rest.is_empty() && rest.len() < 100;
            }
        }
    }

    // Pattern 2: structural keyword followed by a number/letter identifier
    let lower_first = first_word.to_lowercase();
    if matches!(lower_first.as_str(), "chapter" | "part" | "section" | "appendix") {
        let rest = trimmed.splitn(2, char::is_whitespace).nth(1).unwrap_or("");
        if !rest.is_empty() {
            return true;
        }
    }

    false
}

/// Estimate heading level (1–6) from section numbering depth and font-size ratio.
///
/// The hierarchy is determined primarily by structural cues:
///   - Explicit numbering depth: `1` → level 1, `1.2` → level 2, `1.2.3` → level 3.
///   - Font-size ratio: the bigger the heading relative to body text, the
///     higher the level.
fn guess_heading_level(text: &str, size_ratio: f64) -> u32 {
    let first_word = text.split_whitespace().next().unwrap_or("");
    let cleaned = first_word.trim_end_matches('.');

    // Numbered sections: "1" = level 1, "1.2" = level 2, "1.2.3" = level 3
    if !cleaned.is_empty() && cleaned.chars().all(|c| c.is_ascii_digit() || c == '.') {
        let dots = cleaned.chars().filter(|&c| c == '.').count();
        return match dots {
            0 => 1,
            1 => 2,
            _ => (dots as u32 + 1).min(6),
        };
    }

    // Letter-number section numbering: "A" = 1, "A.1" = 2, "A.1.2" = 3
    if !cleaned.is_empty() {
        let mut chars = cleaned.chars();
        if let Some(first) = chars.next() {
            if first.is_ascii_uppercase() {
                let rest: String = chars.collect();
                if rest.is_empty() || rest.chars().all(|c| c.is_ascii_digit() || c == '.') {
                    let dots = rest.chars().filter(|&c| c == '.').count();
                    return ((1 + dots) as u32).min(6);
                }
            }
        }
    }

    // Fall back to font-size ratio
    if size_ratio >= 1.8 {
        1
    } else if size_ratio >= 1.4 {
        2
    } else if size_ratio >= 1.2 {
        3
    } else {
        2
    }
}

// ---------------------------------------------------------------------------
// Multi-column reordering using positioned text
// ---------------------------------------------------------------------------

fn reorder_multicolumn_text(
    positioned_texts: &[PositionedText],
    page_height: f64,
    boundary: f64,
) -> String {
    let mut left_lines: Vec<(f64, String)> = Vec::new();
    let mut right_lines: Vec<(f64, String)> = Vec::new();

    let tol = 5.0;
    let mut sorted: Vec<&PositionedText> = positioned_texts.iter().collect();
    sorted.sort_by(|a, b| {
        a.y.partial_cmp(&b.y)
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
            .then(a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
    });

    struct Line {
        y: f64,
        x_center: f64,
        text: String,
    }

    let mut lines: Vec<Line> = Vec::new();
    for pt in &sorted {
        if pt.text.trim().is_empty() {
            continue;
        }
        let top_y = page_height - pt.y;
        if let Some(last) = lines.last_mut() {
            if (top_y - last.y).abs() < tol {
                if !last.text.is_empty() && !last.text.ends_with(' ') && !pt.text.starts_with(' ') {
                    last.text.push(' ');
                }
                last.text.push_str(&pt.text);
                continue;
            }
        }
        lines.push(Line {
            y: top_y,
            x_center: pt.x,
            text: pt.text.clone(),
        });
    }

    for line in &lines {
        if line.x_center < boundary {
            left_lines.push((line.y, line.text.clone()));
        } else {
            right_lines.push((line.y, line.text.clone()));
        }
    }

    left_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    right_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = String::new();

    emit_column_lines(&left_lines, &mut result);
    result.push_str("\n\n");
    emit_column_lines(&right_lines, &mut result);

    result
}

fn emit_column_lines(lines: &[(f64, String)], out: &mut String) {
    let mut prev_y = f64::MIN;
    for (y, text) in lines {
        let gap = y - prev_y;
        if gap > 20.0 && prev_y > f64::MIN {
            out.push_str("\n\n");
        } else if prev_y > f64::MIN {
            // Join lines that end with a hyphen followed by a lowercase start
            if out.ends_with('-') {
                let next_start = text.chars().next();
                if next_start.is_some_and(|c| c.is_lowercase()) {
                    out.pop(); // remove trailing hyphen
                    out.push_str(text);
                    prev_y = *y;
                    continue;
                }
            }
            out.push('\n');
        }
        out.push_str(text);
        prev_y = *y;
    }
}

// ---------------------------------------------------------------------------
// Page header/footer detection
// ---------------------------------------------------------------------------

/// Detect repeated header/footer text across pages.
///
/// We check the first few and last few lines on each page and look for
/// short text that appears on at least 30% of the pages (minimum 2).
/// This catches page numbers, running titles, and copyright lines without
/// relying on keyword matching.
fn detect_page_furniture(
    page_texts: &[(u32, &str)],
) -> (Vec<String>, Vec<String>) {
    if page_texts.len() < 2 {
        return (Vec::new(), Vec::new());
    }

    const MAX_LINE_LEN: usize = 120;
    const HEADER_LINES: usize = 3;
    const FOOTER_LINES: usize = 3;

    let mut header_counts: HashMap<String, usize> = HashMap::new();
    let mut footer_counts: HashMap<String, usize> = HashMap::new();

    for (_, text) in page_texts {
        let lines: Vec<&str> = text.lines().collect();

        let mut seen_header: HashSet<String> = HashSet::new();
        for line in lines.iter().take(HEADER_LINES) {
            let trimmed = line.trim().to_string();
            if !trimmed.is_empty() && trimmed.len() < MAX_LINE_LEN && seen_header.insert(trimmed.clone()) {
                *header_counts.entry(trimmed).or_insert(0) += 1;
            }
        }

        let mut seen_footer: HashSet<String> = HashSet::new();
        let footer_start = lines.len().saturating_sub(FOOTER_LINES);
        for line in lines.iter().skip(footer_start) {
            let trimmed = line.trim().to_string();
            if !trimmed.is_empty() && trimmed.len() < MAX_LINE_LEN && seen_footer.insert(trimmed.clone()) {
                *footer_counts.entry(trimmed).or_insert(0) += 1;
            }
        }
    }

    let total = page_texts.len();
    let min_pages = ((total as f64 * 0.3).ceil() as usize).max(2);

    let headers: Vec<String> = header_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_pages)
        .map(|(text, _)| text)
        .collect();
    let footers: Vec<String> = footer_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_pages)
        .map(|(text, _)| text)
        .collect();

    (headers, footers)
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn obj_as_f64(obj: &Object) -> Option<f64> {
    match obj {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(f) => Some(*f as f64),
        _ => None,
    }
}

fn resolve_page_size(doc: &Document, page_id: ObjectId) -> Option<(f64, f64)> {
    if let Some(dims) = try_box_from_dict(doc, page_id, b"CropBox") {
        return Some(dims);
    }
    if let Some(dims) = try_box_from_dict(doc, page_id, b"MediaBox") {
        return Some(dims);
    }
    let mut current_id = page_id;
    for _ in 0..20 {
        let dict = match doc.get_object(current_id).ok().and_then(|o| o.as_dict().ok()) {
            Some(d) => d,
            None => break,
        };
        let parent_ref = match dict.get(b"Parent").ok() {
            Some(obj) => match obj.as_reference() {
                Ok(r) => r,
                Err(_) => break,
            },
            None => break,
        };
        if let Some(dims) = try_box_from_dict(doc, parent_ref, b"MediaBox") {
            return Some(dims);
        }
        current_id = parent_ref;
    }
    None
}

fn try_box_from_dict(doc: &Document, obj_id: ObjectId, key: &[u8]) -> Option<(f64, f64)> {
    let dict = doc.get_object(obj_id).ok()?.as_dict().ok()?;
    let arr = dict.get(key).ok().and_then(|obj| resolve_array(doc, obj))?;
    if arr.len() >= 4 {
        let x0 = obj_as_f64(&arr[0]).unwrap_or(0.0);
        let y0 = obj_as_f64(&arr[1]).unwrap_or(0.0);
        let x1 = obj_as_f64(&arr[2]).unwrap_or(612.0);
        let y1 = obj_as_f64(&arr[3]).unwrap_or(792.0);
        Some(((x1 - x0).abs(), (y1 - y0).abs()))
    } else {
        None
    }
}

fn resolve_array(doc: &Document, obj: &Object) -> Option<Vec<Object>> {
    match obj {
        Object::Array(arr) => Some(arr.clone()),
        Object::Reference(r) => doc.get_object(*r).ok().and_then(|o| o.as_array().ok().cloned()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Backend implementation
// ---------------------------------------------------------------------------

pub struct PdfBackend;

impl Backend for PdfBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let data = std::fs::read(path)
            .with_context(|| format!("Failed to read PDF file: {}", path.display()))?;
        let hash = compute_hash(&data);
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let name = doc_name_from_path(path);
        let mut doc = DoclingDocument::new(&name, filename, InputFormat::Pdf.mimetype(), hash);

        let pdf_doc = Document::load_mem(&data)
            .with_context(|| format!("Failed to parse PDF: {}", path.display()))?;

        let pages = pdf_doc.get_pages();
        let mut page_numbers: Vec<u32> = pages.keys().copied().collect();
        page_numbers.sort();

        // Phase 1: Analyze layout and extract text for all pages
        struct PageData {
            page_num: u32,
            width: f64,
            height: f64,
            raw_text: String,
            layout: Option<LayoutInfo>,
            column_layout: ColumnLayout,
        }

        let mut all_pages: Vec<PageData> = Vec::new();
        let mut all_font_sizes: Vec<f64> = Vec::new();

        for &page_num in &page_numbers {
            let page_id = pages[&page_num];
            let (width, height) = resolve_page_size(&pdf_doc, page_id).unwrap_or((612.0, 792.0));

            doc.pages.insert(
                page_num.to_string(),
                PageItem {
                    size: Size { width, height },
                    page_no: page_num,
                    image: None,
                },
            );

            let raw_text = pdf_doc.extract_text(&[page_num]).unwrap_or_default();
            let layout = analyze_page_layout(&pdf_doc, page_id).ok();

            let column_layout = if let Some(ref li) = layout {
                all_font_sizes.extend(li.text_events.iter().map(|e| e.font_size));
                detect_columns(&li.text_events, width)
            } else {
                ColumnLayout::Single
            };

            all_pages.push(PageData {
                page_num,
                width,
                height,
                raw_text,
                layout,
                column_layout,
            });
        }

        let body_font_size = median_f64(&all_font_sizes);

        // Phase 2: Detect page headers/footers
        let page_text_refs: Vec<(u32, &str)> = all_pages
            .iter()
            .map(|p| (p.page_num, p.raw_text.as_str()))
            .collect();
        let (header_texts, footer_texts) = detect_page_furniture(&page_text_refs);

        // Phase 3: Process each page
        for page_data in &all_pages {
            let text = sanitize_text(&page_data.raw_text);
            let text = text.trim();
            if text.is_empty() {
                // Still check for images - use pdfium for better figure extraction
                if let Some(ref layout) = page_data.layout {
                    extract_page_figures_pdfium(
                        path,
                        layout,
                        page_data.page_num,
                        page_data.width,
                        page_data.height,
                        &mut doc,
                    );
                }
                continue;
            }

            // For multi-column pages, use positioned text for reordering
            let processed_text = match page_data.column_layout {
                ColumnLayout::TwoColumn { boundary } => {
                    if let Some(ref layout) = page_data.layout {
                        if !layout.positioned_texts.is_empty() {
                            let reordered = reorder_multicolumn_text(
                                &layout.positioned_texts,
                                page_data.height,
                                boundary,
                            );
                            sanitize_text(&reordered)
                        } else {
                            text.to_string()
                        }
                    } else {
                        text.to_string()
                    }
                }
                ColumnLayout::Single => text.to_string(),
            };

            // Build font size map for the page
            let fs_info = page_data
                .layout
                .as_ref()
                .map(|li| analyze_font_sizes(&li.text_events));

            // Pre-classify all paragraphs to enable list grouping
            let raw_paragraphs = split_paragraphs(&processed_text);
            let paragraphs = merge_fragment_paragraphs(&raw_paragraphs);
            let mut classified: Vec<(String, DocItemLabel, Option<f64>)> = Vec::new();
            let mut para_idx = 0;

            for paragraph in &paragraphs {
                let trimmed = paragraph.trim();
                if trimmed.is_empty() {
                    continue;
                }

                // Skip page headers/footers - only filter if the paragraph is MOSTLY furniture
                // (i.e., the furniture text is a large portion of the paragraph)
                let is_header = header_texts.iter().any(|h| {
                    let h_trimmed = h.trim();
                    h_trimmed.len() > 5 && trimmed.contains(h_trimmed)
                });
                let is_footer = footer_texts.iter().any(|f| {
                    let f_trimmed = f.trim();
                    f_trimmed.len() > 5 && trimmed.contains(f_trimmed)
                });
                if is_header || is_footer {
                    // Only filter if the furniture text comprises most of the paragraph
                    let furniture_len = header_texts.iter()
                        .chain(footer_texts.iter())
                        .filter(|t| trimmed.contains(t.as_str()))
                        .map(|t| t.len())
                        .max()
                        .unwrap_or(0);
                    
                    // Filter only if furniture is >80% of paragraph length
                    if furniture_len as f64 > trimmed.len() as f64 * 0.8 && trimmed.len() < 150 {
                        let label = if is_footer && !is_header {
                            DocItemLabel::PageFooter
                        } else {
                            DocItemLabel::PageHeader
                        };
                        doc.add_furniture_text(label, trimmed);
                        para_idx += 1;
                        continue;
                    }
                }

                let local_fs = fs_info.as_ref().and_then(|info| {
                    if para_idx < info.sizes_per_event.len() {
                        Some(info.sizes_per_event[para_idx])
                    } else {
                        None
                    }
                });

                let label = classify_paragraph(trimmed, body_font_size, local_fs);
                classified.push((trimmed.to_string(), label, local_fs));
                para_idx += 1;
            }

            // Emit classified paragraphs with list grouping
            let mut current_list_group: Option<String> = None;
            let mut i = 0;
            while i < classified.len() {
                let (ref text, ref label, local_fs) = classified[i];
                let trimmed = text.as_str();

                if *label == DocItemLabel::ListItem {
                    if current_list_group.is_none() {
                        let enumerated = looks_like_numbered_list(trimmed);
                        let group_label = if enumerated {
                            GroupLabel::OrderedList
                        } else {
                            GroupLabel::List
                        };
                        let gidx = doc.add_group("list", group_label, None);
                        current_list_group = Some(doc.groups[gidx].self_ref.clone());
                    }

                    let group_ref = current_list_group.as_ref().unwrap();
                    let enumerated = looks_like_numbered_list(trimmed);
                    let marker = extract_list_marker(trimmed);
                    let item_text = strip_list_marker(trimmed);
                    let idx =
                        doc.add_list_item(&item_text, enumerated, marker.as_deref(), group_ref);

                    doc.texts[idx].prov.push(ProvenanceItem {
                        page_no: page_data.page_num,
                        bbox: BoundingBox {
                            l: 0.0,
                            t: 0.0,
                            r: page_data.width,
                            b: page_data.height,
                            coord_origin: Some("TOPLEFT".to_string()),
                        },
                        charspan: Some((0, trimmed.len())),
                    });
                } else {
                    current_list_group = None;

                    let idx = doc.add_text(label.clone(), trimmed, None);

                    doc.texts[idx].prov.push(ProvenanceItem {
                        page_no: page_data.page_num,
                        bbox: BoundingBox {
                            l: 0.0,
                            t: 0.0,
                            r: page_data.width,
                            b: page_data.height,
                            coord_origin: Some("TOPLEFT".to_string()),
                        },
                        charspan: Some((0, trimmed.len())),
                    });

                    if *label == DocItemLabel::SectionHeader {
                        let size_ratio = local_fs
                            .map(|fs| fs / body_font_size)
                            .unwrap_or(1.5);
                        let level = guess_heading_level(trimmed, size_ratio);
                        doc.texts[idx].level = Some(level);
                    }
                }

                i += 1;
            }

            // Emit tables from ruled lines
            if let Some(ref layout) = page_data.layout {
                let table_regions = detect_table_regions(&layout.paths, page_data.height);
                for region in &table_regions {
                    if let Some((grid, num_rows, num_cols)) =
                        build_table_from_texts(&layout.positioned_texts, region, page_data.height)
                    {
                        // Filter out false positives: require at least 40% non-empty cells
                        // and at least 2 columns with content
                        let total_cells = num_rows * num_cols;
                        let non_empty = grid.iter()
                            .flat_map(|row| row.iter())
                            .filter(|c| !c.trim().is_empty())
                            .count();
                        if total_cells == 0 || (non_empty as f64 / total_cells as f64) < 0.35 {
                            continue;
                        }
                        let cols_with_content: usize = (0..num_cols)
                            .filter(|&c| grid.iter().any(|row| !row[c].trim().is_empty()))
                            .count();
                        if cols_with_content < 2 {
                            continue;
                        }

                        let mut cells = Vec::new();
                        for r in 0..num_rows {
                            for c in 0..num_cols {
                                cells.push(TableCell {
                                    text: grid[r][c].clone(),
                                    start_row_offset_idx: r as u32,
                                    end_row_offset_idx: (r + 1) as u32,
                                    start_col_offset_idx: c as u32,
                                    end_col_offset_idx: (c + 1) as u32,
                                    row_span: 1,
                                    col_span: 1,
                                    column_header: r == 0,
                                    row_header: false,
                                    row_section: false,
                                    fillable: false,
                                    formatted_text: None,
                                });
                            }
                        }

                        let table_idx = doc.add_table(
                            cells,
                            num_rows as u32,
                            num_cols as u32,
                            None,
                        );
                        doc.tables[table_idx].prov.push(ProvenanceItem {
                            page_no: page_data.page_num,
                            bbox: BoundingBox {
                                l: region.x_min,
                                t: region.y_min,
                                r: region.x_max,
                                b: region.y_max,
                                coord_origin: Some("TOPLEFT".to_string()),
                            },
                            charspan: None,
                        });
                    }
                }

                // Emit images - use pdfium for better figure extraction
                extract_page_figures_pdfium(
                    path,
                    layout,
                    page_data.page_num,
                    page_data.width,
                    page_data.height,
                    &mut doc,
                );
            }
        }

        Ok(doc)
    }
}

fn extract_image_bytes(
    pdf_doc: &Document,
    stream: &lopdf::Stream,
    obj_id: ObjectId,
) -> (Option<Vec<u8>>, Option<String>) {
    let dict = &stream.dict;

    let filter = dict
        .get(b"Filter")
        .ok()
        .and_then(|f| f.as_name().ok())
        .unwrap_or(b"");

    if filter == b"DCTDecode" {
        // JPEG: raw stream bytes are a valid JPEG
        let bytes = stream.content.clone();
        if !bytes.is_empty() {
            return (Some(bytes), Some("image/jpeg".to_string()));
        }
    }

    if filter == b"FlateDecode" || filter == b"" {
        if let Ok(mut owned_stream) = pdf_doc.get_object(obj_id).and_then(|o| {
            o.as_stream()
                .map(|s| s.clone())
                .map_err(|e| lopdf::Error::from(e))
        }) {
            let _ = owned_stream.decompress();
            let raw = &owned_stream.content;
            let img_width = dict
                .get(b"Width")
                .ok()
                .and_then(|o| o.as_i64().ok())
                .unwrap_or(0) as u32;
            let img_height = dict
                .get(b"Height")
                .ok()
                .and_then(|o| o.as_i64().ok())
                .unwrap_or(0) as u32;
            let bpc = dict
                .get(b"BitsPerComponent")
                .ok()
                .and_then(|o| o.as_i64().ok())
                .unwrap_or(8) as u32;

            if img_width > 0 && img_height > 0 && bpc == 8 {
                let cs = dict
                    .get(b"ColorSpace")
                    .ok()
                    .and_then(|o| o.as_name().ok())
                    .unwrap_or(b"DeviceRGB");
                let channels: u32 = if cs == b"DeviceGray" { 1 } else { 3 };
                let expected = (img_width * img_height * channels) as usize;

                if raw.len() >= expected {
                    let png_bytes = encode_raw_as_png(raw, img_width, img_height, channels);
                    if let Some(bytes) = png_bytes {
                        return (Some(bytes), Some("image/png".to_string()));
                    }
                }
            }
        }
    }

    if filter == b"JPXDecode" {
        let bytes = stream.content.clone();
        if !bytes.is_empty() {
            return (Some(bytes), Some("image/jp2".to_string()));
        }
    }

    (None, None)
}

fn encode_raw_as_png(raw: &[u8], width: u32, height: u32, channels: u32) -> Option<Vec<u8>> {
    use image::{DynamicImage, GrayImage, RgbImage};
    use std::io::Cursor;

    let img: DynamicImage = if channels == 1 {
        let buf = GrayImage::from_raw(width, height, raw[..(width * height) as usize].to_vec())?;
        DynamicImage::ImageLuma8(buf)
    } else {
        let buf = RgbImage::from_raw(
            width,
            height,
            raw[..(width * height * 3) as usize].to_vec(),
        )?;
        DynamicImage::ImageRgb8(buf)
    };

    let mut png_buf = Cursor::new(Vec::new());
    img.write_to(&mut png_buf, image::ImageFormat::Png).ok()?;
    Some(png_buf.into_inner())
}

/// Render PDF pages using pdfium and extract image objects as figure regions.
/// This provides much better figure extraction than extracting raw embedded images.
fn render_page_figures(
    pdf_path: &Path,
    page_num: u32,
    page_width: f64,
    page_height: f64,
    image_bboxes: &[(f64, f64, f64, f64)], // (x, y, width, height) in PDF coords
) -> Vec<(Vec<u8>, f64, f64, f64, f64)> {
    // Return: Vec of (jpeg_bytes, x, y, width, height)
    let mut results = Vec::new();
    
    // Try to bind to pdfium in multiple locations:
    // 1. Relative to executable (for bundled distribution)
    // 2. ./lib directory (development)
    // 3. Current directory
    // 4. System library
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));
    
    let mut binding_attempts: Vec<Result<Box<dyn PdfiumLibraryBindings>, PdfiumError>> = Vec::new();
    
    // Try relative to executable first
    if let Some(ref exe_path) = exe_dir {
        let lib_path = exe_path.join("lib");
        binding_attempts.push(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(lib_path.to_str().unwrap_or("./lib")))
        );
        binding_attempts.push(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(exe_path.to_str().unwrap_or("./")))
        );
    }
    
    // Try common development paths
    binding_attempts.push(Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./lib")));
    binding_attempts.push(Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./")));
    binding_attempts.push(Pdfium::bind_to_system_library());
    
    let bindings = binding_attempts
        .into_iter()
        .find_map(|r| r.ok());
    
    let bindings = match bindings {
        Some(b) => b,
        None => {
            log::warn!("Failed to bind to pdfium: library not found in any location");
            return results;
        }
    };
    
    let pdfium = Pdfium::new(bindings);
    
    let document = match pdfium.load_pdf_from_file(pdf_path, None) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("Failed to load PDF with pdfium: {:?}", e);
            return results;
        }
    };
    
    // Get the page (0-indexed in pdfium)
    let page_index = page_num.saturating_sub(1) as u16;
    let page = match document.pages().get(page_index) {
        Ok(p) => p,
        Err(_) => return results,
    };
    
    // Render the full page at 2x scale for better quality
    let scale = 2.0;
    let render_config = PdfRenderConfig::new()
        .set_target_width((page_width * scale) as i32)
        .set_maximum_height((page_height * scale) as i32);
    
    let page_bitmap = match page.render_with_config(&render_config) {
        Ok(b) => b,
        Err(_) => return results,
    };
    
    let page_image = page_bitmap.as_image();
    let rendered_width = page_image.width() as f64;
    let rendered_height = page_image.height() as f64;
    
    // Scale factors from PDF coordinates to rendered pixels
    let scale_x = rendered_width / page_width;
    let scale_y = rendered_height / page_height;
    
    // Crop each figure region
    for &(x, y, w, h) in image_bboxes {
        // Convert PDF coordinates to pixel coordinates
        // PDF origin is bottom-left, image origin is top-left
        let px_x = (x * scale_x) as u32;
        let px_y = ((page_height - y - h) * scale_y) as u32; // Flip Y
        let px_w = (w * scale_x) as u32;
        let px_h = (h * scale_y) as u32;
        
        // Bounds check
        if px_x + px_w > page_image.width() || px_y + px_h > page_image.height() {
            continue;
        }
        if px_w < 50 || px_h < 50 {
            continue; // Skip very small regions
        }
        
        // Crop the region
        let cropped = page_image.crop_imm(px_x, px_y, px_w, px_h);
        
        // Resize if exceeds max dimension (1536px)
        const MAX_DIM: u32 = 1536;
        let (final_w, final_h) = if px_w > MAX_DIM || px_h > MAX_DIM {
            let ratio = (MAX_DIM as f32 / px_w as f32).min(MAX_DIM as f32 / px_h as f32);
            ((px_w as f32 * ratio) as u32, (px_h as f32 * ratio) as u32)
        } else {
            (px_w, px_h)
        };
        
        let resized = if final_w != px_w || final_h != px_h {
            cropped.resize(final_w, final_h, image::imageops::FilterType::Lanczos3)
        } else {
            cropped
        };
        
        let rgb_img = resized.to_rgb8();
        
        // Compress with decreasing quality until size target (500KB) is met
        const MAX_SIZE_KB: usize = 500;
        const QUALITY_START: u8 = 95;
        const QUALITY_MIN: u8 = 30;
        const QUALITY_STEP: u8 = 10;
        
        let max_size_bytes = MAX_SIZE_KB * 1024;
        let mut quality = QUALITY_START;
        let mut jpeg_bytes: Vec<u8>;
        
        loop {
            jpeg_bytes = Vec::new();
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                std::io::Cursor::new(&mut jpeg_bytes),
                quality,
            );
            if encoder.encode_image(&rgb_img).is_err() {
                break;
            }
            
            if jpeg_bytes.len() <= max_size_bytes || quality <= QUALITY_MIN {
                break;
            }
            quality = quality.saturating_sub(QUALITY_STEP);
        }
        
        if !jpeg_bytes.is_empty() {
            log::info!("  PDF figure: {}x{} -> {}x{}, {}KB, q={}", 
                       px_w, px_h, final_w, final_h, jpeg_bytes.len() / 1024, quality);
            results.push((jpeg_bytes, x, y, w, h));
        }
    }
    
    results
}

/// Check if bbox `a` contains or significantly overlaps with bbox `b`
fn bbox_contains_or_overlaps(a: &(f64, f64, f64, f64), b: &(f64, f64, f64, f64)) -> bool {
    let (ax, ay, aw, ah) = *a;
    let (bx, by, bw, bh) = *b;
    
    // Check if `a` contains `b` (with some tolerance)
    let tolerance = 20.0;
    let a_contains_b = bx >= ax - tolerance && 
                       by >= ay - tolerance &&
                       bx + bw <= ax + aw + tolerance &&
                       by + bh <= ay + ah + tolerance;
    
    if a_contains_b && (aw * ah) > (bw * bh) {
        return true;
    }
    
    // Check for significant overlap (>70% of smaller bbox)
    let overlap_x = (ax + aw).min(bx + bw) - ax.max(bx);
    let overlap_y = (ay + ah).min(by + bh) - ay.max(by);
    
    if overlap_x > 0.0 && overlap_y > 0.0 {
        let overlap_area = overlap_x * overlap_y;
        let smaller_area = (aw * ah).min(bw * bh);
        if overlap_area > smaller_area * 0.7 && (aw * ah) > (bw * bh) {
            return true;
        }
    }
    
    false
}

/// Filter out nested/overlapping bounding boxes, keeping only the larger ones
fn filter_overlapping_bboxes(bboxes: Vec<(f64, f64, f64, f64)>) -> Vec<(f64, f64, f64, f64)> {
    if bboxes.len() <= 1 {
        return bboxes;
    }
    
    let mut result = Vec::new();
    let mut skip_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
    
    // Sort by area (largest first)
    let mut sorted: Vec<(usize, (f64, f64, f64, f64))> = bboxes.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| {
        let area_a = a.1.2 * a.1.3;
        let area_b = b.1.2 * b.1.3;
        area_b.partial_cmp(&area_a).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    for i in 0..sorted.len() {
        if skip_indices.contains(&sorted[i].0) {
            continue;
        }
        
        let bbox_i = &sorted[i].1;
        result.push(*bbox_i);
        
        // Mark smaller overlapping boxes to skip
        for j in (i + 1)..sorted.len() {
            if !skip_indices.contains(&sorted[j].0) {
                let bbox_j = &sorted[j].1;
                if bbox_contains_or_overlaps(bbox_i, bbox_j) {
                    skip_indices.insert(sorted[j].0);
                }
            }
        }
    }
    
    result
}

/// Extract figure regions from a PDF page by rendering and cropping image bounding boxes.
/// Falls back to embedded image extraction if pdfium is not available.
fn extract_page_figures_pdfium(
    pdf_path: &Path,
    layout: &LayoutInfo,
    page_num: u32,
    page_width: f64,
    page_height: f64,
    doc: &mut DoclingDocument,
) {
    // Collect image bounding boxes from layout, filtering small images
    let image_bboxes: Vec<(f64, f64, f64, f64)> = layout
        .images
        .iter()
        .filter(|img| img.width > 100.0 && img.height > 100.0) // Filter small images
        .map(|img| (img.x, img.y, img.width, img.height))
        .collect();
    
    // Filter out nested/overlapping bounding boxes
    let image_bboxes = filter_overlapping_bboxes(image_bboxes);
    
    if image_bboxes.is_empty() {
        return;
    }
    
    let rendered_figures = render_page_figures(
        pdf_path,
        page_num,
        page_width,
        page_height,
        &image_bboxes,
    );
    
    for (jpeg_bytes, x, y, w, h) in rendered_figures {
        let img_y_top = page_height - y - h;
        let idx = doc.add_picture(None, None);
        doc.pictures[idx].prov.push(ProvenanceItem {
            page_no: page_num,
            bbox: BoundingBox {
                l: x,
                t: img_y_top,
                r: x + w,
                b: img_y_top + h,
                coord_origin: Some("TOPLEFT".to_string()),
            },
            charspan: None,
        });
        
        // Load image to get actual pixel dimensions
        let (px_w, px_h) = image::load_from_memory(&jpeg_bytes)
            .map(|i| (i.width() as f64, i.height() as f64))
            .unwrap_or((w, h));
        
        let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg_bytes);
        let uri = format!("data:image/jpeg;base64,{}", b64);
        doc.set_picture_image(
            idx,
            ImageRef {
                mimetype: "image/jpeg".to_string(),
                dpi: 144, // 2x scale
                size: ImageSize {
                    width: px_w,
                    height: px_h,
                },
                uri,
            },
        );
    }
}

fn emit_images(doc: &mut DoclingDocument, layout: &LayoutInfo, page_num: u32, page_height: f64) {
    for img in &layout.images {
        let img_y_top = page_height - img.y;
        let idx = doc.add_picture(None, None);
        doc.pictures[idx].prov.push(ProvenanceItem {
            page_no: page_num,
            bbox: BoundingBox {
                l: img.x,
                t: img_y_top,
                r: img.x + img.width,
                b: img_y_top + img.height,
                coord_origin: Some("TOPLEFT".to_string()),
            },
            charspan: None,
        });

        if let (Some(ref data), Some(ref mimetype)) = (&img.data, &img.mimetype) {
            let (px_w, px_h) = image::load_from_memory(data)
                .map(|i| (i.width() as f64, i.height() as f64))
                .unwrap_or((img.width, img.height));
            let b64 = base64::engine::general_purpose::STANDARD.encode(data);
            let uri = format!("data:{};base64,{}", mimetype, b64);
            doc.set_picture_image(
                idx,
                ImageRef {
                    mimetype: mimetype.clone(),
                    dpi: 72,
                    size: ImageSize {
                        width: px_w,
                        height: px_h,
                    },
                    uri,
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_paragraphs_double_newline() {
        let text = "Hello world\n\nSecond paragraph";
        let result = split_paragraphs(text);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "Hello world");
        assert_eq!(result[1], "Second paragraph");
    }

    #[test]
    fn test_split_paragraphs_empty() {
        let text = "  \n\n  ";
        let result = split_paragraphs(text);
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_paragraphs_single_block() {
        let text = "One long paragraph that wraps across\nmultiple lines but is really\njust one paragraph.";
        let result = split_paragraphs(text);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_looks_like_section_header() {
        assert!(looks_like_section_header("1 Introduction"));
        assert!(looks_like_section_header("1.2 Methods"));
        assert!(looks_like_section_header("5.1 Hyper Parameter Optimization"));
        assert!(looks_like_section_header("A.1 Appendix"));
        assert!(looks_like_section_header("Chapter 3"));
        assert!(looks_like_section_header("Part II"));
        assert!(looks_like_section_header("Section 1.2 Overview"));
        assert!(looks_like_section_header("Appendix A"));
        // Pure content words are NOT structural headers without numbering
        assert!(!looks_like_section_header("References"));
        assert!(!looks_like_section_header("Abstract"));
        assert!(!looks_like_section_header("Introduction"));
        assert!(!looks_like_section_header(
            "This is a long sentence that explains something in detail and is clearly not a heading."
        ));
        assert!(!looks_like_section_header(""));
    }

    #[test]
    fn test_looks_like_list_item() {
        assert!(looks_like_list_item("• First item"));
        assert!(looks_like_list_item("- Second item"));
        assert!(looks_like_list_item("1. Third item"));
        assert!(looks_like_list_item("(a) Fourth item"));
        assert!(!looks_like_list_item("Normal paragraph text."));
    }

    #[test]
    fn test_sanitize_text() {
        assert_eq!(sanitize_text("e\u{FB03}cient"), "efficient");
        assert_eq!(sanitize_text("some-\nword"), "someword");
        assert_eq!(sanitize_text("Some-\nThing"), "Some-\nThing");
    }

    #[test]
    fn test_guess_heading_level() {
        // Numbered sections derive level from numbering depth
        assert_eq!(guess_heading_level("1 Introduction", 1.5), 1);
        assert_eq!(guess_heading_level("1.2 Sub-section", 1.5), 2);
        assert_eq!(guess_heading_level("1.2.3 Deep", 1.5), 3);
        // Letter-number sections
        assert_eq!(guess_heading_level("A Overview", 1.5), 1);
        assert_eq!(guess_heading_level("A.1 Details", 1.5), 2);
        // Non-numbered headings fall back to font-size ratio
        assert_eq!(guess_heading_level("Abstract", 1.5), 2); // ratio 1.5 → level 2
        assert_eq!(guess_heading_level("Abstract", 1.8), 1); // ratio 1.8 → level 1
        assert_eq!(guess_heading_level("Abstract", 1.2), 3); // ratio 1.2 → level 3
    }

    #[test]
    fn test_classify_paragraph() {
        assert_eq!(
            classify_paragraph("5.1 Hyper Parameter Optimization", 12.0, None),
            DocItemLabel::SectionHeader
        );
        assert_eq!(
            classify_paragraph(
                "This is a normal paragraph with some text content that explains things.",
                12.0,
                None
            ),
            DocItemLabel::Text
        );
    }

    #[test]
    fn test_obj_as_f64() {
        assert_eq!(obj_as_f64(&Object::Integer(42)), Some(42.0));
        let real_val = obj_as_f64(&Object::Real(3.14)).unwrap();
        assert!((real_val - 3.14).abs() < 0.001);
        assert!(obj_as_f64(&Object::Boolean(true)).is_none());
    }

    #[test]
    fn test_median_f64() {
        assert_eq!(median_f64(&[10.0, 12.0, 12.0, 14.0, 24.0]), 12.0);
        assert_eq!(median_f64(&[]), 12.0);
        assert_eq!(median_f64(&[9.0]), 9.0);
    }

    #[test]
    fn test_path_segment_orientation() {
        let h = PathSegment { x1: 0.0, y1: 100.0, x2: 200.0, y2: 100.5 };
        assert!(h.is_horizontal(1.0));
        assert!(!h.is_vertical(1.0));

        let v = PathSegment { x1: 100.0, y1: 0.0, x2: 100.5, y2: 200.0 };
        assert!(!v.is_horizontal(1.0));
        assert!(v.is_vertical(1.0));
    }

    #[test]
    fn test_ctm_multiply() {
        let identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let translate = [1.0, 0.0, 0.0, 1.0, 100.0, 200.0];
        let result = ctm_multiply(&translate, &identity);
        assert!((result[4] - 100.0).abs() < 0.001);
        assert!((result[5] - 200.0).abs() < 0.001);
    }
}
