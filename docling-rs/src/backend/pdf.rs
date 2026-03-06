use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use anyhow::Context;
use base64::Engine;

use crate::models::common::{DocItemLabel, GroupLabel, InputFormat};
use crate::models::document::{compute_hash, doc_name_from_path, DoclingDocument};
use crate::models::page::{BoundingBox, PageItem, ProvenanceItem, Size};
use crate::models::picture::{ImageRef, ImageSize};
use crate::models::table::TableCell;

use super::Backend;

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

const BULLET_GLYPHS: &[char] = &['•', '○', '■', '□', '◦', '▪'];

// ---------------------------------------------------------------------------
// pdf_oxide integration types
// ---------------------------------------------------------------------------

#[cfg(feature = "pdf-oxide")]
use pdf_oxide::PdfDocument;

#[derive(Debug, Clone)]
struct AssembledBlock {
    text: String,
    bbox: (f64, f64, f64, f64), // l, t, r, b in TOPLEFT coords
    font_size: f64,
    is_artifact: bool,
}

// ---------------------------------------------------------------------------
// pdf_oxide text assembly pipeline
// ---------------------------------------------------------------------------

#[cfg(feature = "pdf-oxide")]
fn assemble_page_blocks(
    oxide_doc: &mut PdfDocument,
    page_index: usize,
    page_height: f64,
) -> Vec<AssembledBlock> {
    use pdf_oxide::pipeline::{
        ReadingOrderConfig, ReadingOrderContext, ReadingOrderStrategyType, TextPipeline,
        TextPipelineConfig,
    };

    let spans = match oxide_doc.extract_spans(page_index) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    if spans.is_empty() {
        return Vec::new();
    }

    let config = TextPipelineConfig {
        reading_order: ReadingOrderConfig {
            strategy: ReadingOrderStrategyType::XYCut,
        },
        ..Default::default()
    };

    let pipeline = TextPipeline::with_config(config);
    let context = ReadingOrderContext::default();

    let ordered = match pipeline.process(spans, context) {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };

    if ordered.is_empty() {
        return Vec::new();
    }

    group_ordered_spans_into_blocks(&ordered, page_height)
}

#[cfg(feature = "pdf-oxide")]
fn group_ordered_spans_into_blocks(
    ordered: &[pdf_oxide::pipeline::OrderedTextSpan],
    page_height: f64,
) -> Vec<AssembledBlock> {
    let mut blocks: Vec<AssembledBlock> = Vec::new();

    let mut current_text = String::new();
    let mut current_l = f64::MAX;
    let mut current_t = f64::MAX;
    let mut current_r = f64::MIN;
    let mut current_b = f64::MIN;
    let mut current_font_size: f64 = 0.0;
    let mut current_font_count: usize = 0;
    let mut current_is_artifact = false;
    let mut prev_bottom: Option<f64> = None;
    let mut prev_font_size: f64 = 0.0;

    for ospan in ordered {
        let span = &ospan.span;
        let text = span.text.trim();
        if text.is_empty() {
            continue;
        }

        let is_artifact = span.artifact_type.is_some();

        let bbox = &span.bbox;
        let span_l = bbox.x as f64;
        let span_t = page_height - (bbox.y + bbox.height) as f64;
        let span_r = (bbox.x + bbox.width) as f64;
        let span_b = page_height - bbox.y as f64;
        let span_fs = span.font_size as f64;
        let line_height = (span_b - span_t).abs().max(span_fs);

        let should_break = if let Some(pb) = prev_bottom {
            let gap = (span_t - pb).abs();
            let font_change = (span_fs - prev_font_size).abs() / prev_font_size.max(1.0);
            let artifact_change = is_artifact != current_is_artifact;
            gap > line_height * 1.2 || font_change > 0.25 || artifact_change
        } else {
            false
        };

        if should_break && !current_text.is_empty() {
            blocks.push(AssembledBlock {
                text: current_text.trim().to_string(),
                bbox: (current_l, current_t, current_r, current_b),
                font_size: if current_font_count > 0 {
                    current_font_size / current_font_count as f64
                } else {
                    12.0
                },
                is_artifact: current_is_artifact,
            });
            current_text = String::new();
            current_l = f64::MAX;
            current_t = f64::MAX;
            current_r = f64::MIN;
            current_b = f64::MIN;
            current_font_size = 0.0;
            current_font_count = 0;
        }

        if !current_text.is_empty() {
            current_text.push(' ');
        }
        current_text.push_str(text);
        current_l = current_l.min(span_l);
        current_t = current_t.min(span_t);
        current_r = current_r.max(span_r);
        current_b = current_b.max(span_b);
        current_font_size += span_fs;
        current_font_count += 1;
        current_is_artifact = is_artifact;
        prev_bottom = Some(span_b);
        prev_font_size = span_fs;
    }

    if !current_text.trim().is_empty() {
        blocks.push(AssembledBlock {
            text: current_text.trim().to_string(),
            bbox: (current_l, current_t, current_r, current_b),
            font_size: if current_font_count > 0 {
                current_font_size / current_font_count as f64
            } else {
                12.0
            },
            is_artifact: current_is_artifact,
        });
    }

    blocks
}

// ---------------------------------------------------------------------------
// lopdf path extraction (kept for table detection)
// ---------------------------------------------------------------------------

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

fn extract_paths_from_page(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> Vec<PathSegment> {
    let content_data = match doc.get_page_content(page_id) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let ops = match lopdf::content::Content::decode(&content_data) {
        Ok(c) => c.operations,
        Err(_) => return Vec::new(),
    };

    let mut paths = Vec::new();
    let mut gstate_stack: Vec<GState> = vec![GState::default()];
    let mut current_gstate = GState::default();
    let mut path_start: Option<(f64, f64)> = None;
    let mut current_point: Option<(f64, f64)> = None;
    let mut subpath_segments: Vec<PathSegment> = Vec::new();

    for op in &ops {
        let opname = op.operator.as_str();
        let operands = &op.operands;

        match opname {
            "q" => {
                gstate_stack.push(current_gstate.clone());
            }
            "Q" => {
                if let Some(gs) = gstate_stack.pop() {
                    current_gstate = gs;
                }
            }
            "cm" => {
                if operands.len() >= 6 {
                    let vals: Vec<f64> = operands.iter().filter_map(obj_as_f64).collect();
                    if vals.len() == 6 {
                        let new_ctm = [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]];
                        current_gstate.ctm = ctm_multiply(&new_ctm, &current_gstate.ctm);
                    }
                }
            }
            "m" => {
                if operands.len() >= 2 {
                    if let (Some(x), Some(y)) = (obj_as_f64(&operands[0]), obj_as_f64(&operands[1]))
                    {
                        let (tx, ty) = ctm_transform(&current_gstate.ctm, x, y);
                        path_start = Some((tx, ty));
                        current_point = Some((tx, ty));
                    }
                }
            }
            "l" => {
                if operands.len() >= 2 {
                    if let (Some(x), Some(y)) = (obj_as_f64(&operands[0]), obj_as_f64(&operands[1]))
                    {
                        let (tx, ty) = ctm_transform(&current_gstate.ctm, x, y);
                        if let Some((cx, cy)) = current_point {
                            subpath_segments.push(PathSegment {
                                x1: cx,
                                y1: cy,
                                x2: tx,
                                y2: ty,
                            });
                        }
                        current_point = Some((tx, ty));
                    }
                }
            }
            "re" => {
                if operands.len() >= 4 {
                    let vals: Vec<f64> = operands.iter().filter_map(obj_as_f64).collect();
                    if vals.len() == 4 {
                        let (x0, y0) = ctm_transform(&current_gstate.ctm, vals[0], vals[1]);
                        let (x1, y1) = ctm_transform(
                            &current_gstate.ctm,
                            vals[0] + vals[2],
                            vals[1] + vals[3],
                        );
                        subpath_segments.push(PathSegment {
                            x1: x0,
                            y1: y0,
                            x2: x1,
                            y2: y0,
                        });
                        subpath_segments.push(PathSegment {
                            x1,
                            y1: y0,
                            x2: x1,
                            y2: y1,
                        });
                        subpath_segments.push(PathSegment {
                            x1,
                            y1,
                            x2: x0,
                            y2: y1,
                        });
                        subpath_segments.push(PathSegment {
                            x1: x0,
                            y1,
                            x2: x0,
                            y2: y0,
                        });
                        current_point = Some((x0, y0));
                        path_start = Some((x0, y0));
                    }
                }
            }
            "h" => {
                if let (Some(cp), Some(ps)) = (current_point, path_start) {
                    if (cp.0 - ps.0).abs() > 0.01 || (cp.1 - ps.1).abs() > 0.01 {
                        subpath_segments.push(PathSegment {
                            x1: cp.0,
                            y1: cp.1,
                            x2: ps.0,
                            y2: ps.1,
                        });
                    }
                    current_point = path_start;
                }
            }
            "S" | "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" | "n" => {
                paths.append(&mut subpath_segments);
                path_start = None;
                current_point = None;
            }
            _ => {}
        }
    }

    paths
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

    if h_lines.len() < 4 || v_lines.len() < 3 {
        return Vec::new();
    }

    h_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    v_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let x_range = h_lines
        .iter()
        .fold((f64::MAX, f64::MIN), |(min, max), (_, xs, xe)| {
            (min.min(*xs), max.max(*xe))
        });
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
            .is_none_or(|&last: &f64| (v - last).abs() > tol)
        {
            result.push(v);
        }
    }
    result
}

fn build_table_from_blocks(
    blocks: &[AssembledBlock],
    region: &TableRegion,
) -> Option<(Vec<Vec<String>>, usize, usize)> {
    let num_rows = region.h_lines.len().saturating_sub(1);
    let num_cols = region.v_lines.len().saturating_sub(1);
    if num_rows == 0 || num_cols == 0 {
        return None;
    }

    let mut grid: Vec<Vec<String>> = vec![vec![String::new(); num_cols]; num_rows];

    for block in blocks {
        let (bl, bt, _br, _bb) = block.bbox;
        if bt < region.y_min - 5.0 || bt > region.y_max + 5.0 {
            continue;
        }
        if bl < region.x_min - 5.0 || bl > region.x_max + 5.0 {
            continue;
        }

        let col = region
            .v_lines
            .windows(2)
            .position(|w| bl >= w[0] - 5.0 && bl <= w[1] + 5.0)
            .unwrap_or(0)
            .min(num_cols - 1);
        let row = region
            .h_lines
            .windows(2)
            .position(|w| bt >= w[0] - 5.0 && bt <= w[1] + 5.0)
            .unwrap_or(0)
            .min(num_rows - 1);

        if !grid[row][col].is_empty() {
            grid[row][col].push(' ');
        }
        grid[row][col].push_str(block.text.trim());
    }

    Some((grid, num_rows, num_cols))
}

// ---------------------------------------------------------------------------
// Text processing and classification
// ---------------------------------------------------------------------------

fn sanitize_text(text: &str) -> String {
    let mut result = text.to_string();

    result = result
        .replace('\u{FB00}', "ff")
        .replace('\u{FB01}', "fi")
        .replace('\u{FB02}', "fl")
        .replace('\u{FB03}', "ffi")
        .replace('\u{FB04}', "ffl")
        .replace(['\u{FB05}', '\u{FB06}'], "st");

    result = result
        .replace('\u{2044}', "/")
        .replace(['\u{2019}', '\u{2018}'], "'")
        .replace(['\u{201C}', '\u{201D}'], "\"")
        .replace('\u{00A0}', " ");

    let mut out = String::with_capacity(result.len());
    let chars: Vec<char> = result.chars().collect();
    let mut i = 0;
    while i < chars.len() {
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

fn median_f64(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 12.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
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

    if looks_like_list_item(trimmed) {
        return DocItemLabel::ListItem;
    }

    if looks_like_section_header(trimmed) {
        return DocItemLabel::SectionHeader;
    }

    if let Some(fs) = local_font_size {
        let ratio = fs / body_font_size;
        if ratio >= 1.3 && trimmed.len() < 200 && trimmed.lines().count() <= 3 {
            return DocItemLabel::SectionHeader;
        }
    }

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
                if ratio >= 1.25 {
                    return DocItemLabel::SectionHeader;
                }
            }
        }
    }

    if let Some(fs) = local_font_size {
        let ratio = fs / body_font_size;
        if ratio < 0.85
            && trimmed.len() < 300
            && (trimmed.starts_with(|c: char| c.is_ascii_digit())
                || trimmed.starts_with('*')
                || trimmed.starts_with('†'))
        {
            return DocItemLabel::Footnote;
        }
    }

    DocItemLabel::Text
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

    if cleaned.chars().all(|c| c.is_ascii_digit() || c == '.')
        && cleaned.chars().any(|c| c.is_ascii_digit())
    {
        let rest = trimmed
            .split_once(char::is_whitespace)
            .map(|x| x.1)
            .unwrap_or("");
        return !rest.is_empty() && rest.len() < 100;
    }

    let mut chars = cleaned.chars();
    if let Some(first_ch) = chars.next() {
        if first_ch.is_ascii_uppercase() {
            let rest_str: String = chars.collect();
            let is_section_number =
                rest_str.is_empty() || rest_str.chars().all(|c| c.is_ascii_digit() || c == '.');
            if is_section_number {
                let rest = trimmed
                    .split_once(char::is_whitespace)
                    .map(|x| x.1)
                    .unwrap_or("");
                return !rest.is_empty() && rest.len() < 100;
            }
        }
    }

    let lower_first = first_word.to_lowercase();
    if matches!(
        lower_first.as_str(),
        "chapter" | "part" | "section" | "appendix"
    ) {
        let rest = trimmed
            .split_once(char::is_whitespace)
            .map(|x| x.1)
            .unwrap_or("");
        if !rest.is_empty() {
            return true;
        }
    }

    false
}

fn guess_heading_level(text: &str, size_ratio: f64) -> u32 {
    let first_word = text.split_whitespace().next().unwrap_or("");
    let cleaned = first_word.trim_end_matches('.');

    if !cleaned.is_empty() && cleaned.chars().all(|c| c.is_ascii_digit() || c == '.') {
        let dots = cleaned.chars().filter(|&c| c == '.').count();
        return match dots {
            0 => 1,
            1 => 2,
            _ => (dots as u32 + 1).min(6),
        };
    }

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

fn starts_with_bullet(text: &str) -> bool {
    if let Some(first) = text.chars().next() {
        if BULLET_GLYPHS.contains(&first) {
            let rest = &text[first.len_utf8()..];
            return rest.starts_with(' ') || rest.is_empty();
        }
        if (first == '-' || first == '–' || first == '—') && text.len() > 2 {
            return text[first.len_utf8()..].starts_with(' ');
        }
    }
    false
}

fn starts_with_ordered_marker(text: &str) -> Option<usize> {
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();

    if !bytes.is_empty() && bytes[0].is_ascii_digit() {
        if let Some(pos) = trimmed.find(|c: char| !c.is_ascii_digit()) {
            let after = &trimmed[pos..];
            if after.starts_with(". ") || after.starts_with(") ") {
                return Some(pos + 1);
            }
        }
    }
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

fn strip_list_marker(text: &str) -> String {
    let trimmed = text.trim();

    if let Some(first) = trimmed.chars().next() {
        if BULLET_GLYPHS.contains(&first) || first == '-' || first == '–' || first == '—' {
            let rest = &trimmed[first.len_utf8()..];
            if let Some(stripped) = rest.strip_prefix(' ') {
                return stripped.trim_start().to_string();
            }
        }
    }

    if let Some(marker_end) = starts_with_ordered_marker(trimmed) {
        let after = &trimmed[marker_end..];
        return after.trim_start().to_string();
    }

    trimmed.to_string()
}

// ---------------------------------------------------------------------------
// Page header/footer detection
// ---------------------------------------------------------------------------

fn detect_page_furniture_from_blocks(
    page_blocks: &[(u32, Vec<AssembledBlock>)],
) -> (Vec<String>, Vec<String>) {
    if page_blocks.len() < 2 {
        return (Vec::new(), Vec::new());
    }

    const MAX_LINE_LEN: usize = 120;
    const HEADER_BLOCKS: usize = 2;
    const FOOTER_BLOCKS: usize = 2;

    let mut header_counts: HashMap<String, usize> = HashMap::new();
    let mut footer_counts: HashMap<String, usize> = HashMap::new();

    for (_page_num, blocks) in page_blocks {
        let non_artifact: Vec<&AssembledBlock> = blocks.iter().filter(|b| !b.is_artifact).collect();

        for block in non_artifact.iter().take(HEADER_BLOCKS) {
            let text = block.text.trim().to_string();
            if !text.is_empty() && text.len() < MAX_LINE_LEN {
                *header_counts.entry(text).or_insert(0) += 1;
            }
        }

        for block in non_artifact.iter().rev().take(FOOTER_BLOCKS) {
            let text = block.text.trim().to_string();
            if !text.is_empty() && text.len() < MAX_LINE_LEN {
                *footer_counts.entry(text).or_insert(0) += 1;
            }
        }
    }

    let threshold = (page_blocks.len() as f64 * 0.3).ceil() as usize;
    let min_pages = 2;

    let headers: Vec<String> = header_counts
        .into_iter()
        .filter(|(_, count)| *count >= threshold && *count >= min_pages)
        .map(|(text, _)| text)
        .collect();

    let footers: Vec<String> = footer_counts
        .into_iter()
        .filter(|(_, count)| *count >= threshold && *count >= min_pages)
        .map(|(text, _)| text)
        .collect();

    (headers, footers)
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn obj_as_f64(obj: &lopdf::Object) -> Option<f64> {
    match obj {
        lopdf::Object::Integer(i) => Some(*i as f64),
        lopdf::Object::Real(f) => Some(*f as f64),
        _ => None,
    }
}

#[cfg(feature = "pdf-oxide")]
fn resolve_page_size_oxide(oxide_doc: &mut PdfDocument, page_index: usize) -> (f64, f64) {
    match oxide_doc.get_page_media_box(page_index) {
        Ok((x0, y0, x1, y1)) => ((x1 as f64 - x0 as f64).abs(), (y1 as f64 - y0 as f64).abs()),
        Err(_) => (612.0, 792.0),
    }
}

#[allow(dead_code)]
fn resolve_page_size_lopdf(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> Option<(f64, f64)> {
    if let Some(dims) = try_box_from_dict(doc, page_id, b"CropBox") {
        return Some(dims);
    }
    if let Some(dims) = try_box_from_dict(doc, page_id, b"MediaBox") {
        return Some(dims);
    }
    let mut current_id = page_id;
    for _ in 0..20 {
        let dict = match doc
            .get_object(current_id)
            .ok()
            .and_then(|o| o.as_dict().ok())
        {
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

#[allow(dead_code)]
fn try_box_from_dict(
    doc: &lopdf::Document,
    obj_id: lopdf::ObjectId,
    key: &[u8],
) -> Option<(f64, f64)> {
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

#[allow(dead_code)]
fn resolve_array(doc: &lopdf::Document, obj: &lopdf::Object) -> Option<Vec<lopdf::Object>> {
    match obj {
        lopdf::Object::Array(arr) => Some(arr.clone()),
        lopdf::Object::Reference(r) => doc
            .get_object(*r)
            .ok()
            .and_then(|o| o.as_array().ok().cloned()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Page rendering via pdfium (for vector diagram extraction)
// ---------------------------------------------------------------------------

#[cfg(feature = "pdfium-render")]
fn detect_diagram_regions(
    blocks: &[AssembledBlock],
    page_width: f64,
    page_height: f64,
) -> Vec<(f64, f64, f64, f64)> {
    if page_width <= 0.0 || page_height <= 0.0 {
        return Vec::new();
    }

    let content_blocks: Vec<&AssembledBlock> = blocks.iter().filter(|b| !b.is_artifact).collect();
    if content_blocks.is_empty() {
        return Vec::new();
    }

    let min_gap_height = page_height * 0.12;
    let min_region_width = page_width * 0.20;

    let mut occupied: Vec<(f64, f64)> = content_blocks
        .iter()
        .map(|b| {
            let (_, t, _, b_coord) = b.bbox;
            (t.min(b_coord), t.max(b_coord))
        })
        .collect();
    occupied.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (top, bot) in &occupied {
        if let Some(last) = merged.last_mut() {
            if *top <= last.1 + 2.0 {
                last.1 = last.1.max(*bot);
                continue;
            }
        }
        merged.push((*top, *bot));
    }

    let mut regions: Vec<(f64, f64, f64, f64)> = Vec::new();

    // Gap before first text block
    if let Some(first) = merged.first() {
        if first.0 > min_gap_height {
            regions.push((0.0, 0.0, page_width, first.0));
        }
    }

    // Gaps between text blocks
    for window in merged.windows(2) {
        let gap_top = window[0].1;
        let gap_bottom = window[1].0;
        let gap_height = gap_bottom - gap_top;
        if gap_height > min_gap_height {
            let margin = 2.0;
            regions.push((0.0, gap_top + margin, page_width, gap_bottom - margin));
        }
    }

    // Gap after last text block
    if let Some(last) = merged.last() {
        let remaining = page_height - last.1;
        if remaining > min_gap_height {
            regions.push((0.0, last.1, page_width, page_height));
        }
    }

    regions
        .into_iter()
        .filter(|(l, t, r, b)| {
            let w = (r - l).abs();
            let h = (b - t).abs();
            w >= min_region_width && h >= min_gap_height
        })
        .collect()
}

#[cfg(feature = "pdfium-render")]
fn region_has_visual_content(img: &image::DynamicImage) -> bool {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    if w < 10 || h < 10 {
        return false;
    }

    // Reject very elongated strips (likely header/footer background areas)
    let aspect = w as f64 / h as f64;
    if aspect > 5.0 || aspect < 0.2 {
        return false;
    }

    // Edge density: count adjacent pixel pairs with sharp transitions.
    // Smooth gradients/backgrounds have very few sharp edges; real diagrams have many.
    let step_x = (w / 60).max(1);
    let step_y = (h / 60).max(1);
    let edge_threshold: i32 = 40;
    let mut edge_count: u64 = 0;
    let mut sample_count: u64 = 0;

    let mut y = 1;
    while y < h {
        let mut x = 1;
        while x < w {
            let p = rgb.get_pixel(x, y);
            let px_left = rgb.get_pixel(x - 1, y);
            let px_up = rgb.get_pixel(x, y - 1);

            let diff_h = (p[0] as i32 - px_left[0] as i32).abs()
                + (p[1] as i32 - px_left[1] as i32).abs()
                + (p[2] as i32 - px_left[2] as i32).abs();
            let diff_v = (p[0] as i32 - px_up[0] as i32).abs()
                + (p[1] as i32 - px_up[1] as i32).abs()
                + (p[2] as i32 - px_up[2] as i32).abs();

            if diff_h > edge_threshold || diff_v > edge_threshold {
                edge_count += 1;
            }
            sample_count += 1;
            x += step_x;
        }
        y += step_y;
    }

    if sample_count == 0 {
        return false;
    }

    let edge_ratio = edge_count as f64 / sample_count as f64;
    // Diagrams/screenshots typically have >5% of sampled pixels on an edge;
    // smooth gradients and solid fills are well under 3%.
    edge_ratio > 0.03
}

#[cfg(feature = "pdfium-render")]
fn bbox_overlap_ratio(
    a: (f64, f64, f64, f64),
    b: (f64, f64, f64, f64),
) -> f64 {
    let (al, at, ar, ab) = a;
    let (bl, bt, br, bb) = b;
    let ol = al.max(bl);
    let ot = at.max(bt);
    let or_ = ar.min(br);
    let ob = ab.min(bb);
    if ol >= or_ || ot >= ob {
        return 0.0;
    }
    let overlap_area = (or_ - ol) * (ob - ot);
    let area_a = (ar - al).abs() * (ab - at).abs();
    let area_b = (br - bl).abs() * (bb - bt).abs();
    let smaller = area_a.min(area_b);
    if smaller <= 0.0 {
        return 0.0;
    }
    overlap_area / smaller
}

#[cfg(feature = "pdfium-render")]
fn emit_rendered_diagrams(
    doc: &mut DoclingDocument,
    pdfium: &pdfium_render::prelude::Pdfium,
    pdf_bytes: &[u8],
    page_index: usize,
    page_num: u32,
    page_width: f64,
    page_height: f64,
    blocks: &[AssembledBlock],
    raster_bboxes: &[(f64, f64, f64, f64)],
    seen_image_hashes: &mut HashSet<u64>,
) {
    use pdfium_render::prelude::*;

    let regions = detect_diagram_regions(blocks, page_width, page_height);
    if regions.is_empty() {
        return;
    }

    let pdfium_doc = match pdfium.load_pdf_from_byte_slice(pdf_bytes, None) {
        Ok(d) => d,
        Err(_) => return,
    };

    let page = match pdfium_doc.pages().get(page_index as u16) {
        Ok(p) => p,
        Err(_) => return,
    };

    let render_dpi: f64 = 200.0;
    let scale = render_dpi / 72.0;

    for (reg_l, reg_t, reg_r, reg_b) in &regions {
        // Skip if this region substantially overlaps with an already-extracted raster image
        let region_bbox = (*reg_l, *reg_t, *reg_r, *reg_b);
        let overlaps_raster = raster_bboxes
            .iter()
            .any(|rb| bbox_overlap_ratio(region_bbox, *rb) > 0.50);
        if overlaps_raster {
            continue;
        }

        // Convert TOPLEFT coords to PDF bottom-left coords for crop box
        let pdf_left = *reg_l;
        let pdf_bottom = page_height - *reg_b;
        let pdf_right = *reg_r;
        let pdf_top = page_height - *reg_t;

        let crop_w = ((pdf_right - pdf_left) * scale).round() as i32;
        let crop_h = ((pdf_top - pdf_bottom) * scale).round() as i32;

        if crop_w < 20 || crop_h < 20 {
            continue;
        }

        // Render full page and crop the region
        let full_w = (page_width * scale).round() as i32;
        let config = PdfRenderConfig::new()
            .set_target_width(full_w)
            .set_maximum_height(full_w * 4);

        let bitmap = match page.render_with_config(&config) {
            Ok(b) => b,
            Err(_) => continue,
        };

        let full_img: image::DynamicImage = bitmap.as_image();

        // Crop the region from the full rendered page
        let px_l = ((*reg_l) * scale).round() as u32;
        let px_t = ((*reg_t) * scale).round() as u32;
        let px_w = crop_w as u32;
        let px_h = crop_h as u32;

        let (full_w, full_h) = (full_img.width(), full_img.height());
        if px_l + px_w > full_w || px_t + px_h > full_h {
            continue;
        }

        let cropped = full_img.crop_imm(px_l, px_t, px_w, px_h);

        if !region_has_visual_content(&cropped) {
            continue;
        }

        // Downscale if needed
        let max_dim = cropped.width().max(cropped.height());
        let final_img = if max_dim > 2048 {
            cropped.resize(2048, 2048, image::imageops::FilterType::Lanczos3)
        } else {
            cropped
        };

        let mut png_buf = std::io::Cursor::new(Vec::new());
        if final_img
            .write_to(&mut png_buf, image::ImageFormat::Png)
            .is_err()
        {
            continue;
        }
        let png_bytes = png_buf.into_inner();

        // Content-hash dedup
        let content_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            final_img.width().hash(&mut hasher);
            final_img.height().hash(&mut hasher);
            let sample_len = png_bytes.len().min(4096);
            png_bytes[..sample_len].hash(&mut hasher);
            png_bytes.len().hash(&mut hasher);
            hasher.finish()
        };
        if !seen_image_hashes.insert(content_hash) {
            continue;
        }

        let idx = doc.add_picture(None, None);
        doc.pictures[idx].prov.push(ProvenanceItem {
            page_no: page_num,
            bbox: BoundingBox {
                l: *reg_l,
                t: *reg_t,
                r: *reg_r,
                b: *reg_b,
                coord_origin: Some("TOPLEFT".to_string()),
            },
            charspan: None,
        });

        let b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
        let uri = format!("data:image/png;base64,{}", b64);
        doc.set_picture_image(
            idx,
            ImageRef {
                mimetype: "image/png".to_string(),
                dpi: render_dpi as u32,
                size: ImageSize {
                    width: final_img.width() as f64,
                    height: final_img.height() as f64,
                },
                uri,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Image utilities
// ---------------------------------------------------------------------------

fn downscale_png(png_bytes: &[u8], orig_w: u32, orig_h: u32, max_dim: u32) -> Option<(Vec<u8>, u32, u32)> {
    use image::DynamicImage;
    use std::io::Cursor;

    let img = image::load_from_memory(png_bytes).ok()?;
    let scale = max_dim as f64 / orig_w.max(orig_h) as f64;
    let new_w = ((orig_w as f64) * scale).round() as u32;
    let new_h = ((orig_h as f64) * scale).round() as u32;
    if new_w == 0 || new_h == 0 {
        return None;
    }
    let resized: DynamicImage = img.resize(new_w, new_h, image::imageops::FilterType::Lanczos3);
    let mut buf = Cursor::new(Vec::new());
    resized.write_to(&mut buf, image::ImageFormat::Png).ok()?;
    Some((buf.into_inner(), new_w, new_h))
}

// ---------------------------------------------------------------------------
// Image extraction via pdf_oxide
// ---------------------------------------------------------------------------

#[cfg(feature = "pdf-oxide")]
fn emit_images_oxide(
    doc: &mut DoclingDocument,
    oxide_doc: &mut PdfDocument,
    page_index: usize,
    page_num: u32,
    page_width: f64,
    page_height: f64,
    seen_image_hashes: &mut HashSet<u64>,
) -> Vec<(f64, f64, f64, f64)> {
    let images = match oxide_doc.extract_images(page_index) {
        Ok(imgs) => imgs,
        Err(_) => return Vec::new(),
    };

    let mut emitted_bboxes: Vec<(f64, f64, f64, f64)> = Vec::new();

    let page_area = page_width * page_height;

    for img in &images {
        let (img_l, img_t, img_r, img_b) = if let Some(bbox) = img.bbox() {
            (
                bbox.x as f64,
                page_height - (bbox.y + bbox.height) as f64,
                (bbox.x + bbox.width) as f64,
                page_height - bbox.y as f64,
            )
        } else {
            continue;
        };

        let img_w = (img_r - img_l).abs();
        let img_h = (img_b - img_t).abs();

        // Skip images covering >80% of the page (slide backgrounds)
        if page_area > 0.0 && (img_w * img_h) > page_area * 0.80 {
            continue;
        }

        // Skip tiny decorative images (< 30x30 in page coords)
        if img_w < 30.0 && img_h < 30.0 {
            continue;
        }

        // Skip images with pixel dimensions > 4000 on any side (raw backgrounds)
        let px_w = img.width();
        let px_h = img.height();
        if px_w > 4000 || px_h > 4000 {
            continue;
        }

        // Skip very small pixel images (likely single-color fills or tiny glyphs)
        if px_w < 4 || px_h < 4 {
            continue;
        }

        let png_bytes = match img.to_png_bytes() {
            Ok(b) => b,
            Err(_) => continue,
        };

        // Skip near-uniform-color images (solid fills, masks, gradients).
        // A meaningful image compresses to at least ~0.05 bytes per pixel as PNG.
        // Solid-color or near-solid images compress to far less.
        let pixel_count = (px_w as u64) * (px_h as u64);
        if pixel_count > 1000 {
            let bytes_per_pixel = png_bytes.len() as f64 / pixel_count as f64;
            if bytes_per_pixel < 0.05 {
                continue;
            }
        }

        // Content-hash dedup: skip images we've already seen
        let content_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            px_w.hash(&mut hasher);
            px_h.hash(&mut hasher);
            let sample_len = png_bytes.len().min(4096);
            png_bytes[..sample_len].hash(&mut hasher);
            png_bytes.len().hash(&mut hasher);
            hasher.finish()
        };
        if !seen_image_hashes.insert(content_hash) {
            continue;
        }

        // Downscale oversized images to cap at 2048px on longest side
        let max_dim = px_w.max(px_h);
        let (final_bytes, final_w, final_h) = if max_dim > 2048 {
            match downscale_png(&png_bytes, px_w, px_h, 2048) {
                Some((b, w, h)) => (b, w, h),
                None => (png_bytes, px_w, px_h),
            }
        } else {
            (png_bytes, px_w, px_h)
        };

        let idx = doc.add_picture(None, None);
        doc.pictures[idx].prov.push(ProvenanceItem {
            page_no: page_num,
            bbox: BoundingBox {
                l: img_l,
                t: img_t,
                r: img_r,
                b: img_b,
                coord_origin: Some("TOPLEFT".to_string()),
            },
            charspan: None,
        });

        let b64 = base64::engine::general_purpose::STANDARD.encode(&final_bytes);
        let uri = format!("data:image/png;base64,{}", b64);
        doc.set_picture_image(
            idx,
            ImageRef {
                mimetype: "image/png".to_string(),
                dpi: 72,
                size: ImageSize {
                    width: final_w as f64,
                    height: final_h as f64,
                },
                uri,
            },
        );

        emitted_bboxes.push((img_l, img_t, img_r, img_b));
    }

    emitted_bboxes
}

// ---------------------------------------------------------------------------
// Backend implementation
// ---------------------------------------------------------------------------

pub struct PdfBackend;

#[cfg(feature = "pdf-oxide")]
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

        let mut oxide_doc = PdfDocument::from_bytes(data.clone())
            .with_context(|| format!("pdf_oxide failed to parse PDF: {}", path.display()))?;

        let page_count = oxide_doc.page_count().unwrap_or(0);
        if page_count == 0 {
            return Ok(doc);
        }

        // Also load with lopdf for path-based table detection
        let lopdf_doc = lopdf::Document::load_mem(&data).ok();
        let lopdf_pages: BTreeMap<u32, lopdf::ObjectId> = lopdf_doc
            .as_ref()
            .map(|d| d.get_pages())
            .unwrap_or_default();

        // Phase 1: Extract blocks and collect font sizes for all pages
        #[allow(dead_code)]
        struct PageData {
            page_num: u32,
            page_index: usize,
            width: f64,
            height: f64,
            blocks: Vec<AssembledBlock>,
        }

        let mut all_pages: Vec<PageData> = Vec::new();
        let mut all_font_sizes: Vec<f64> = Vec::new();

        for page_index in 0..page_count {
            let page_num = (page_index + 1) as u32;
            let (width, height) = resolve_page_size_oxide(&mut oxide_doc, page_index);

            doc.pages.insert(
                page_num.to_string(),
                PageItem {
                    size: Size { width, height },
                    page_no: page_num,
                    image: None,
                },
            );

            let blocks = assemble_page_blocks(&mut oxide_doc, page_index, height);

            for block in &blocks {
                if !block.is_artifact {
                    all_font_sizes.push(block.font_size);
                }
            }

            all_pages.push(PageData {
                page_num,
                page_index,
                width,
                height,
                blocks,
            });
        }

        let body_font_size = median_f64(&all_font_sizes);

        // Phase 2: Detect page furniture
        let page_block_refs: Vec<(u32, Vec<AssembledBlock>)> = all_pages
            .iter()
            .map(|p| (p.page_num, p.blocks.clone()))
            .collect();
        let (header_texts, footer_texts) = detect_page_furniture_from_blocks(&page_block_refs);

        // Initialize pdfium for vector diagram rendering (auto-downloads binary if needed)
        #[cfg(feature = "pdfium-render")]
        let pdfium_instance: Option<pdfium_render::prelude::Pdfium> =
            pdfium_auto::bind_pdfium_silent()
                .or_else(|_| pdfium_auto::bind_pdfium(None))
                .ok();

        // Phase 3: Classify and emit
        let mut seen_image_hashes: HashSet<u64> = HashSet::new();
        for page_data in &all_pages {
            let content_blocks: Vec<&AssembledBlock> =
                page_data.blocks.iter().filter(|b| !b.is_artifact).collect();

            if content_blocks.is_empty() {
                let raster_bboxes = emit_images_oxide(
                    &mut doc,
                    &mut oxide_doc,
                    page_data.page_index,
                    page_data.page_num,
                    page_data.width,
                    page_data.height,
                    &mut seen_image_hashes,
                );

                #[cfg(feature = "pdfium-render")]
                if let Some(ref pdfium) = pdfium_instance {
                    emit_rendered_diagrams(
                        &mut doc,
                        pdfium,
                        &data,
                        page_data.page_index,
                        page_data.page_num,
                        page_data.width,
                        page_data.height,
                        &page_data.blocks,
                        &raster_bboxes,
                        &mut seen_image_hashes,
                    );
                }

                continue;
            }

            let mut classified: Vec<(&AssembledBlock, DocItemLabel)> = Vec::new();

            for block in &content_blocks {
                let text = sanitize_text(&block.text);
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let is_header = header_texts.iter().any(|h| {
                    let h_trimmed = h.trim();
                    h_trimmed.len() > 5 && trimmed.contains(h_trimmed)
                });
                let is_footer = footer_texts.iter().any(|f| {
                    let f_trimmed = f.trim();
                    f_trimmed.len() > 5 && trimmed.contains(f_trimmed)
                });
                if is_header || is_footer {
                    let furniture_len = header_texts
                        .iter()
                        .chain(footer_texts.iter())
                        .filter(|t| trimmed.contains(t.as_str()))
                        .map(|t| t.len())
                        .max()
                        .unwrap_or(0);
                    if furniture_len as f64 > trimmed.len() as f64 * 0.8
                        && trimmed.len() < 150
                    {
                        let label = if is_footer && !is_header {
                            DocItemLabel::PageFooter
                        } else {
                            DocItemLabel::PageHeader
                        };
                        let fidx = doc.add_furniture_text(label, trimmed);
                        let (bl, bt, br, bb) = block.bbox;
                        doc.texts[fidx].prov.push(ProvenanceItem {
                            page_no: page_data.page_num,
                            bbox: BoundingBox {
                                l: bl,
                                t: bt,
                                r: br,
                                b: bb,
                                coord_origin: Some("TOPLEFT".to_string()),
                            },
                            charspan: Some((0, trimmed.len())),
                        });
                        continue;
                    }
                }

                let label = classify_paragraph(trimmed, body_font_size, Some(block.font_size));
                classified.push((block, label));
            }

            let mut current_list_group: Option<String> = None;
            let mut i = 0;
            while i < classified.len() {
                let (block, ref label) = classified[i];
                let text = sanitize_text(&block.text);
                let trimmed = text.trim();
                let (bl, bt, br, bb) = block.bbox;

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
                            l: bl,
                            t: bt,
                            r: br,
                            b: bb,
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
                            l: bl,
                            t: bt,
                            r: br,
                            b: bb,
                            coord_origin: Some("TOPLEFT".to_string()),
                        },
                        charspan: Some((0, trimmed.len())),
                    });

                    if *label == DocItemLabel::SectionHeader {
                        let size_ratio = block.font_size / body_font_size;
                        let level = guess_heading_level(trimmed, size_ratio);
                        doc.texts[idx].level = Some(level);
                    }
                }

                i += 1;
            }

            // Emit tables from lopdf path segments
            if let Some(ref lopdf_doc) = lopdf_doc {
                let lopdf_page_num = page_data.page_num;
                if let Some(&page_id) = lopdf_pages.get(&lopdf_page_num) {
                    let paths = extract_paths_from_page(lopdf_doc, page_id);
                    let table_regions = detect_table_regions(&paths, page_data.height);
                    for region in &table_regions {
                        if let Some((grid, num_rows, num_cols)) =
                            build_table_from_blocks(&page_data.blocks, region)
                        {
                            let total_cells = num_rows * num_cols;
                            let non_empty = grid
                                .iter()
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
                            for (r, row) in grid.iter().enumerate().take(num_rows) {
                                for (c, cell_text) in row.iter().enumerate().take(num_cols) {
                                    cells.push(TableCell {
                                        text: cell_text.clone(),
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

                            let table_idx =
                                doc.add_table(cells, num_rows as u32, num_cols as u32, None);
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
                }
            }

            // Emit raster images from pdf_oxide
            let raster_bboxes = emit_images_oxide(
                &mut doc,
                &mut oxide_doc,
                page_data.page_index,
                page_data.page_num,
                page_data.width,
                page_data.height,
                &mut seen_image_hashes,
            );

            // Render vector diagram regions via pdfium
            #[cfg(feature = "pdfium-render")]
            if let Some(ref pdfium) = pdfium_instance {
                emit_rendered_diagrams(
                    &mut doc,
                    pdfium,
                    &data,
                    page_data.page_index,
                    page_data.page_num,
                    page_data.width,
                    page_data.height,
                    &page_data.blocks,
                    &raster_bboxes,
                    &mut seen_image_hashes,
                );
            }
        }

        Ok(doc)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_looks_like_section_header() {
        assert!(looks_like_section_header("1 Introduction"));
        assert!(looks_like_section_header("1.2 Methods"));
        assert!(looks_like_section_header(
            "5.1 Hyper Parameter Optimization"
        ));
        assert!(looks_like_section_header("A.1 Appendix"));
        assert!(looks_like_section_header("Chapter 3"));
        assert!(looks_like_section_header("Part II"));
        assert!(looks_like_section_header("Section 1.2 Overview"));
        assert!(looks_like_section_header("Appendix A"));
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
        assert_eq!(guess_heading_level("1 Introduction", 1.5), 1);
        assert_eq!(guess_heading_level("1.2 Sub-section", 1.5), 2);
        assert_eq!(guess_heading_level("1.2.3 Deep", 1.5), 3);
        assert_eq!(guess_heading_level("A Overview", 1.5), 1);
        assert_eq!(guess_heading_level("A.1 Details", 1.5), 2);
        assert_eq!(guess_heading_level("Abstract", 1.5), 2);
        assert_eq!(guess_heading_level("Abstract", 1.8), 1);
        assert_eq!(guess_heading_level("Abstract", 1.2), 3);
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
        assert_eq!(obj_as_f64(&lopdf::Object::Integer(42)), Some(42.0));
        let real_val = obj_as_f64(&lopdf::Object::Real(3.14)).unwrap();
        assert!((real_val - 3.14).abs() < 0.001);
        assert!(obj_as_f64(&lopdf::Object::Boolean(true)).is_none());
    }

    #[test]
    fn test_median_f64() {
        assert_eq!(median_f64(&[10.0, 12.0, 12.0, 14.0, 24.0]), 12.0);
        assert_eq!(median_f64(&[]), 12.0);
        assert_eq!(median_f64(&[9.0]), 9.0);
    }

    #[test]
    fn test_path_segment_orientation() {
        let h = PathSegment {
            x1: 0.0,
            y1: 100.0,
            x2: 200.0,
            y2: 100.5,
        };
        assert!(h.is_horizontal(1.0));
        assert!(!h.is_vertical(1.0));

        let v = PathSegment {
            x1: 100.0,
            y1: 0.0,
            x2: 100.5,
            y2: 200.0,
        };
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
