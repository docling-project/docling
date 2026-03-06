use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use anyhow::Context;
use base64::Engine;
use quick_xml::events::Event;
use quick_xml::Reader;

use crate::models::common::{DocItemLabel, GroupLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::picture::{ImageRef, ImageSize};
use crate::models::table::TableCell;
use crate::models::text::TextFormatting;

use super::Backend;

pub struct DocxBackend;

impl Backend for DocxBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::Docx)?;
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open DOCX file: {}", path.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("Invalid ZIP/DOCX file: {}", path.display()))?;

        let styles = parse_styles(&mut archive);
        let numbering = parse_numbering(&mut archive);
        let rels = parse_relationships(&mut archive, "word/_rels/document.xml.rels");
        let media = preload_media(&mut archive);
        let comments = parse_comments_xml(&mut archive);

        let document_xml = read_zip_entry(&mut archive, "word/document.xml")
            .context("Missing word/document.xml in DOCX")?;
        parse_document_xml(
            &document_xml,
            &styles,
            &numbering,
            &rels,
            &media,
            &comments,
            &mut doc,
        )?;

        for i in 1..=20 {
            let hdr = format!("word/header{}.xml", i);
            if let Ok(xml) = read_zip_entry(&mut archive, &hdr) {
                parse_furniture_xml(&xml, DocItemLabel::PageHeader, &mut doc);
            }
            let ftr = format!("word/footer{}.xml", i);
            if let Ok(xml) = read_zip_entry(&mut archive, &ftr) {
                parse_furniture_xml(&xml, DocItemLabel::PageFooter, &mut doc);
            }
        }

        Ok(doc)
    }
}

// ---------------------------------------------------------------------------
// Relationships
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Relationships {
    map: HashMap<String, (String, bool)>,
}

impl Relationships {
    fn resolve(&self, r_id: &str) -> Option<(&str, bool)> {
        self.map.get(r_id).map(|(t, ext)| (t.as_str(), *ext))
    }

    fn resolve_url(&self, r_id: &str) -> Option<String> {
        self.resolve(r_id)
            .filter(|(_, is_external)| *is_external)
            .map(|(target, _)| target.to_string())
    }

    fn resolve_media_path(&self, r_id: &str) -> Option<String> {
        self.resolve(r_id)
            .filter(|(_, is_external)| !*is_external)
            .map(|(target, _)| {
                if target.starts_with('/') {
                    target.trim_start_matches('/').to_string()
                } else {
                    format!("word/{}", target)
                }
            })
    }
}

fn parse_relationships(archive: &mut zip::ZipArchive<std::fs::File>, path: &str) -> Relationships {
    let mut rels = Relationships::default();
    let xml = match read_zip_entry(archive, path) {
        Ok(x) => x,
        Err(_) => return rels,
    };
    let mut reader = Reader::from_str(&xml);
    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local = get_local_name(e);
                if local == "Relationship" {
                    if let (Some(id), Some(target)) = (get_attr(e, "Id"), get_attr(e, "Target")) {
                        let is_external = get_attr(e, "TargetMode")
                            .is_some_and(|m| m.eq_ignore_ascii_case("External"));
                        rels.map.insert(id, (target, is_external));
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }
    rels
}

// ---------------------------------------------------------------------------
// Media preloading
// ---------------------------------------------------------------------------

fn preload_media(archive: &mut zip::ZipArchive<std::fs::File>) -> HashMap<String, Vec<u8>> {
    let mut media = HashMap::new();
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
        .filter(|n| n.starts_with("word/media/"))
        .collect();
    for name in names {
        if let Ok(mut entry) = archive.by_name(&name) {
            let mut buf = Vec::new();
            if entry.read_to_end(&mut buf).is_ok() {
                media.insert(name, buf);
            }
        }
    }
    media
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Styles {
    heading_styles: HashMap<String, u32>,
    based_on: HashMap<String, String>,
    title_style_ids: Vec<String>,
    /// Style-level numbering: styleId -> (numId, ilvl)
    style_num_pr: HashMap<String, (String, u32)>,
}

impl Styles {
    fn resolve_heading_level(&self, style_id: &str) -> Option<u32> {
        if let Some(&level) = self.heading_styles.get(style_id) {
            return Some(level);
        }
        let mut visited = std::collections::HashSet::new();
        let mut current = style_id.to_string();
        while let Some(parent) = self.based_on.get(&current) {
            if !visited.insert(parent.clone()) {
                break;
            }
            if let Some(&level) = self.heading_styles.get(parent.as_str()) {
                return Some(level);
            }
            current = parent.clone();
        }
        None
    }
}

fn parse_styles(archive: &mut zip::ZipArchive<std::fs::File>) -> Styles {
    let mut styles = Styles::default();
    let xml = match read_zip_entry(archive, "word/styles.xml") {
        Ok(x) => x,
        Err(_) => return styles,
    };

    let mut reader = Reader::from_str(&xml);
    let mut in_style = false;
    let mut current_style_id = String::new();
    let mut current_style_name = String::new();
    let mut in_style_num_pr = false;
    let mut style_num_id = String::new();
    let mut style_ilvl: u32 = 0;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local = get_local_name(e);
                match local.as_str() {
                    "style" => {
                        in_style = true;
                        current_style_id = get_attr(e, "styleId").unwrap_or_default();
                        current_style_name.clear();
                        in_style_num_pr = false;
                        style_num_id.clear();
                        style_ilvl = 0;
                    }
                    "numPr" if in_style => {
                        in_style_num_pr = true;
                    }
                    "numId" if in_style && in_style_num_pr => {
                        style_num_id = get_attr(e, "val").unwrap_or_default();
                    }
                    "ilvl" if in_style && in_style_num_pr => {
                        style_ilvl = get_attr(e, "val").and_then(|v| v.parse().ok()).unwrap_or(0);
                    }
                    "name" if in_style => {
                        current_style_name = get_attr(e, "val").unwrap_or_default();
                    }
                    "basedOn" if in_style => {
                        if let Some(val) = get_attr(e, "val") {
                            styles.based_on.insert(current_style_id.clone(), val);
                        }
                    }
                    "outlineLvl" if in_style => {
                        if let Some(val) = get_attr(e, "val") {
                            if let Ok(level) = val.parse::<u32>() {
                                styles
                                    .heading_styles
                                    .insert(current_style_id.clone(), level + 1);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let local = get_local_name_end(e);
                if local == "numPr" {
                    in_style_num_pr = false;
                } else if local == "style" {
                    in_style = false;
                    if !style_num_id.is_empty() && style_num_id != "0" {
                        styles
                            .style_num_pr
                            .insert(current_style_id.clone(), (style_num_id.clone(), style_ilvl));
                    }
                    let name_lower = current_style_name.to_lowercase();
                    if name_lower == "title"
                        || name_lower == "titre"
                        || name_lower == "titel"
                        || name_lower == "título"
                        || name_lower == "titolo"
                    {
                        styles.title_style_ids.push(current_style_id.clone());
                    }
                    if (name_lower.starts_with("heading")
                        || name_lower.starts_with("titre")
                        || name_lower.starts_with("überschrift")
                        || name_lower.starts_with("título")
                        || name_lower.starts_with("titolo"))
                        && !styles.heading_styles.contains_key(&current_style_id)
                    {
                        let level = current_style_name
                            .chars()
                            .filter(|c| c.is_ascii_digit())
                            .collect::<String>()
                            .parse::<u32>()
                            .unwrap_or(1);
                        styles
                            .heading_styles
                            .insert(current_style_id.clone(), level);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }
    styles
}

// ---------------------------------------------------------------------------
// Numbering
// ---------------------------------------------------------------------------

#[derive(Default)]
struct NumberingInfo {
    definitions: HashMap<(String, u32), (bool, String)>,
}

fn parse_numbering(archive: &mut zip::ZipArchive<std::fs::File>) -> NumberingInfo {
    let mut info = NumberingInfo::default();
    let xml = match read_zip_entry(archive, "word/numbering.xml") {
        Ok(x) => x,
        Err(_) => return info,
    };

    let mut reader = Reader::from_str(&xml);
    let mut abstract_defs: HashMap<String, Vec<(u32, bool, String)>> = HashMap::new();
    let mut num_to_abstract: HashMap<String, String> = HashMap::new();

    let mut in_abstract = false;
    let mut current_abstract_id = String::new();
    let mut in_lvl = false;
    let mut current_ilvl: u32 = 0;
    let mut current_num_fmt = String::new();
    let mut in_num = false;
    let mut current_num_id = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local = get_local_name(e);
                match local.as_str() {
                    "abstractNum" => {
                        in_abstract = true;
                        current_abstract_id = get_attr(e, "abstractNumId").unwrap_or_default();
                    }
                    "lvl" if in_abstract => {
                        in_lvl = true;
                        current_ilvl = get_attr(e, "ilvl")
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0);
                        current_num_fmt.clear();
                    }
                    "numFmt" if in_lvl => {
                        current_num_fmt = get_attr(e, "val").unwrap_or_default();
                    }
                    "num" => {
                        in_num = true;
                        current_num_id = get_attr(e, "numId").unwrap_or_default();
                    }
                    "abstractNumId" if in_num => {
                        if let Some(val) = get_attr(e, "val") {
                            num_to_abstract.insert(current_num_id.clone(), val);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let local = get_local_name_end(e);
                match local.as_str() {
                    "lvl" if in_abstract => {
                        let is_ordered =
                            !matches!(current_num_fmt.as_str(), "bullet" | "none" | "");
                        abstract_defs
                            .entry(current_abstract_id.clone())
                            .or_default()
                            .push((current_ilvl, is_ordered, current_num_fmt.clone()));
                        in_lvl = false;
                    }
                    "abstractNum" => in_abstract = false,
                    "num" => in_num = false,
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    for (num_id, abstract_id) in &num_to_abstract {
        if let Some(levels) = abstract_defs.get(abstract_id) {
            for (ilvl, is_ordered, fmt) in levels {
                info.definitions
                    .insert((num_id.clone(), *ilvl), (*is_ordered, fmt.clone()));
            }
        }
    }
    info
}

// ---------------------------------------------------------------------------
// Comments
// ---------------------------------------------------------------------------

fn parse_comments_xml(archive: &mut zip::ZipArchive<std::fs::File>) -> HashMap<String, String> {
    let mut comments = HashMap::new();
    let xml = match read_zip_entry(archive, "word/comments.xml") {
        Ok(x) => x,
        Err(_) => return comments,
    };
    let mut reader = Reader::from_str(&xml);
    let mut in_comment = false;
    let mut comment_id = String::new();
    let mut comment_text = String::new();
    let mut in_run = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let local = get_local_name(e);
                match local.as_str() {
                    "comment" => {
                        in_comment = true;
                        comment_id = get_attr(e, "id").unwrap_or_default();
                        comment_text.clear();
                    }
                    "r" if in_comment => in_run = true,
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) if in_run && in_comment => {
                comment_text.push_str(&String::from_utf8_lossy(e.as_ref()));
            }
            Ok(Event::GeneralRef(ref e)) if in_run && in_comment => {
                let entity = String::from_utf8_lossy(e.as_ref());
                let ch = match entity.as_ref() {
                    "amp" => "&",
                    "lt" => "<",
                    "gt" => ">",
                    "apos" => "'",
                    "quot" => "\"",
                    _ => "",
                };
                comment_text.push_str(ch);
            }
            Ok(Event::End(ref e)) => {
                let local = get_local_name_end(e);
                match local.as_str() {
                    "comment" => {
                        if !comment_text.trim().is_empty() {
                            comments.insert(comment_id.clone(), comment_text.trim().to_string());
                        }
                        in_comment = false;
                    }
                    "r" => in_run = false,
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }
    comments
}

// ---------------------------------------------------------------------------
// Table state
// ---------------------------------------------------------------------------

/// A paragraph within a table cell, preserving run-level structure.
#[derive(Clone)]
struct CellParagraph {
    runs: Vec<RunSegment>,
    has_num_pr: bool,
    num_id: String,
    num_ilvl: u32,
    style_id: String,
}

/// Rich content accumulated for a single table cell.
struct RichCellContent {
    paragraphs: Vec<CellParagraph>,
    images: Vec<(Option<String>, Option<String>)>,
}

impl RichCellContent {
    fn new() -> Self {
        Self {
            paragraphs: Vec::new(),
            images: Vec::new(),
        }
    }

    fn to_markdown(&self) -> String {
        let mut parts = Vec::new();
        for para in &self.paragraphs {
            let mut para_md = String::new();
            for run in &para.runs {
                let t = run.text.as_str();
                if t.is_empty() {
                    continue;
                }
                let mut frag = String::new();
                if run.fmt.bold {
                    frag.push_str("**");
                }
                if run.fmt.italic {
                    frag.push('*');
                }
                frag.push_str(t);
                if run.fmt.italic {
                    frag.push('*');
                }
                if run.fmt.bold {
                    frag.push_str("**");
                }
                if let Some(ref url) = run.hyperlink {
                    frag = format!("[{}]({})", frag, url);
                }
                para_md.push_str(&frag);
            }
            if !para_md.is_empty() {
                parts.push(para_md);
            }
        }
        parts.join("\n")
    }
}

struct TableState {
    cells: Vec<TableCell>,
    rich_cells: Vec<(u32, u32, u32, RichCellContent)>,
    row: u32,
    col: u32,
    max_cols: u32,
    in_cell: bool,
    cell_text: String,
    cell_content: RichCellContent,
    current_col_span: u32,
    vmerge_continues: bool,
    vmerge_continuations: Vec<(u32, u32)>,
    prev_para_was_list: bool,
}

impl TableState {
    fn new() -> Self {
        Self {
            cells: Vec::new(),
            rich_cells: Vec::new(),
            row: 0,
            col: 0,
            max_cols: 0,
            in_cell: false,
            cell_text: String::new(),
            cell_content: RichCellContent::new(),
            current_col_span: 1,
            vmerge_continues: false,
            vmerge_continuations: Vec::new(),
            prev_para_was_list: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Run formatting tracking
// ---------------------------------------------------------------------------

#[derive(Default, Clone, PartialEq)]
struct RunFmt {
    bold: bool,
    italic: bool,
    underline: bool,
    strikethrough: bool,
    script: Option<String>,
}

impl RunFmt {
    fn is_default(&self) -> bool {
        !self.bold
            && !self.italic
            && !self.underline
            && !self.strikethrough
            && self.script.is_none()
    }

    fn to_text_formatting(&self) -> Option<TextFormatting> {
        if self.is_default() {
            None
        } else {
            Some(TextFormatting {
                bold: self.bold,
                italic: self.italic,
                underline: self.underline,
                strikethrough: self.strikethrough,
                script: self.script.clone(),
            })
        }
    }
}

/// A single run of text with uniform formatting within a paragraph.
#[derive(Clone)]
struct RunSegment {
    text: String,
    fmt: RunFmt,
    hyperlink: Option<String>,
    is_formula: bool,
}

// ---------------------------------------------------------------------------
// List nesting state
// ---------------------------------------------------------------------------

struct ListNesting {
    /// (num_id, ilvl, group_ref, item_counter)
    stack: Vec<(String, u32, String, u32)>,
}

impl ListNesting {
    fn new() -> Self {
        Self { stack: Vec::new() }
    }

    fn get_or_create_group(
        &mut self,
        num_id: &str,
        ilvl: u32,
        _is_ordered: bool,
        doc: &mut DoclingDocument,
        current_parent: Option<&str>,
    ) -> String {
        while let Some((ref id, lvl, _, _)) = self.stack.last() {
            if id != num_id || lvl > &ilvl {
                self.stack.pop();
            } else {
                break;
            }
        }

        if let Some((ref id, lvl, ref gref, _)) = self.stack.last() {
            if id == num_id && *lvl == ilvl {
                return gref.clone();
            }
        }

        let parent_ref = if let Some((_, _, ref parent_gref, _)) = self.stack.last() {
            Some(parent_gref.as_str())
        } else {
            current_parent
        };

        let gidx = doc.add_group("list", GroupLabel::List, parent_ref);
        let gref = format!("#/groups/{}", gidx);
        self.stack.push((num_id.to_string(), ilvl, gref.clone(), 0));
        gref
    }

    fn next_marker(&mut self, is_ordered: bool) -> String {
        if is_ordered {
            if let Some((_, _, _, ref mut counter)) = self.stack.last_mut() {
                *counter += 1;
                return format!("{}.", counter);
            }
        }
        "-".to_string()
    }

    fn reset(&mut self) {
        self.stack.clear();
    }
}

// ---------------------------------------------------------------------------
// Headers / Footers
// ---------------------------------------------------------------------------

fn parse_furniture_xml(xml: &str, label: DocItemLabel, doc: &mut DoclingDocument) {
    let section_name = match label {
        DocItemLabel::PageHeader => "page header",
        DocItemLabel::PageFooter => "page footer",
        _ => "section",
    };

    struct FurnitureRun {
        text: String,
        bold: bool,
        italic: bool,
        underline: bool,
    }

    let mut reader = Reader::from_str(xml);
    let mut in_paragraph = false;
    let mut in_run = false;
    let mut suppress_field: u32 = 0;
    let mut current_run_text = String::new();
    let mut current_bold = false;
    let mut current_italic = false;
    let mut current_underline = false;
    let mut para_runs: Vec<FurnitureRun> = Vec::new();
    let mut all_paragraphs: Vec<Vec<FurnitureRun>> = Vec::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local = get_local_name(e);
                match local.as_str() {
                    "p" => {
                        in_paragraph = true;
                        para_runs.clear();
                        current_run_text.clear();
                        current_bold = false;
                        current_italic = false;
                        current_underline = false;
                    }
                    "r" if in_paragraph => {
                        in_run = true;
                        current_run_text.clear();
                        current_bold = false;
                        current_italic = false;
                        current_underline = false;
                    }
                    "b" if in_run => current_bold = is_toggle_on(e),
                    "i" if in_run => current_italic = is_toggle_on(e),
                    "u" if in_run => current_underline = is_toggle_on(e),
                    "instrText" => suppress_field += 1,
                    "fldChar" => {
                        let fld_type = e.attributes().filter_map(|a| a.ok()).find(|a| {
                            let k = String::from_utf8_lossy(a.key.as_ref());
                            k == "fldCharType" || k.ends_with(":fldCharType")
                        });
                        if let Some(attr) = fld_type {
                            let val = String::from_utf8_lossy(&attr.value);
                            if val == "begin" {
                                suppress_field += 1;
                            } else if val == "end" {
                                suppress_field = suppress_field.saturating_sub(1);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) if in_run && in_paragraph && suppress_field == 0 => {
                current_run_text.push_str(&String::from_utf8_lossy(e.as_ref()));
            }
            Ok(Event::GeneralRef(ref e)) if in_run && in_paragraph && suppress_field == 0 => {
                let entity = String::from_utf8_lossy(e.as_ref());
                let ch = match entity.as_ref() {
                    "amp" => "&",
                    "lt" => "<",
                    "gt" => ">",
                    "apos" => "'",
                    "quot" => "\"",
                    _ => "",
                };
                current_run_text.push_str(ch);
            }
            Ok(Event::End(ref e)) => {
                let local = get_local_name_end(e);
                match local.as_str() {
                    "r" => {
                        if in_run && !current_run_text.is_empty() {
                            para_runs.push(FurnitureRun {
                                text: current_run_text.clone(),
                                bold: current_bold,
                                italic: current_italic,
                                underline: current_underline,
                            });
                        }
                        in_run = false;
                    }
                    "instrText" => suppress_field = suppress_field.saturating_sub(1),
                    "p" => {
                        in_paragraph = false;
                        if !para_runs.is_empty() {
                            let full: String = para_runs.iter().map(|r| r.text.as_str()).collect();
                            if !full.trim().is_empty() {
                                all_paragraphs.push(std::mem::take(&mut para_runs));
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    if all_paragraphs.is_empty() {
        return;
    }

    let section_idx = doc.add_group(section_name, GroupLabel::Section, Some("#/furniture"));
    let section_ref = format!("#/groups/{}", section_idx);

    for para in &all_paragraphs {
        let has_mixed = para.len() >= 2 && {
            let first = &para[0];
            para.iter().skip(1).any(|r| {
                r.bold != first.bold || r.italic != first.italic || r.underline != first.underline
            })
        };

        if has_mixed {
            let inline_idx = doc.add_group("group", GroupLabel::Inline, Some(&section_ref));
            let inline_ref = format!("#/groups/{}", inline_idx);
            for run in para {
                let text = run.text.trim();
                if text.is_empty() {
                    continue;
                }
                let fmt = if run.bold || run.italic || run.underline {
                    Some(TextFormatting {
                        bold: run.bold,
                        italic: run.italic,
                        underline: run.underline,
                        strikethrough: false,
                        script: None,
                    })
                } else {
                    None
                };
                doc.add_text_ext(DocItemLabel::Text, text, Some(&inline_ref), fmt, None);
            }
        } else {
            let full: String = para.iter().map(|r| r.text.as_str()).collect();
            let text = full.trim();
            if !text.is_empty() {
                doc.add_text(DocItemLabel::Text, text, Some(&section_ref));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Elements that should suppress text extraction
// ---------------------------------------------------------------------------

/// Tags whose text content is metadata (EMU values, field codes) not document text.
fn is_suppressed_text_element(local: &str) -> bool {
    matches!(
        local,
        "extent"
            | "positionH"
            | "positionV"
            | "posOffset"
            | "simplePos"
            | "effectExtent"
            | "wrapPolygon"
            | "start"
            | "lineTo"
            | "instrText"
            | "fldChar"
            | "pctWidth"
            | "pctHeight"
            | "sizeRelH"
            | "sizeRelV"
    )
}

/// True when we are inside a field code sequence (between fldChar begin/end).
fn is_field_code_text(text: &str) -> bool {
    let t = text.trim();
    t.contains("INCLUDEPICTURE")
        || t.contains("MERGEFORMATINET")
        || t.contains("MERGEFORMAT")
        || t.starts_with("HYPERLINK")
}

// ---------------------------------------------------------------------------
// Main document parser
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn parse_document_xml(
    xml: &str,
    styles: &Styles,
    numbering: &NumberingInfo,
    rels: &Relationships,
    media: &HashMap<String, Vec<u8>>,
    comments: &HashMap<String, String>,
    doc: &mut DoclingDocument,
) -> anyhow::Result<()> {
    let mut reader = Reader::from_str(xml);
    let mut current_parent: Option<String> = None;
    let mut in_paragraph = false;
    let mut para_style_id = String::new();
    let mut in_run = false;

    let mut table_stack: Vec<TableState> = Vec::new();
    let mut list_nesting = ListNesting::new();

    let mut has_num_pr = false;
    let mut num_id = String::new();
    let mut num_ilvl: u32 = 0;
    let mut in_tc_pr = false;

    // Heading numbering counters: (numId, ilvl) -> current counter
    let mut heading_counters: HashMap<(String, u32), u32> = HashMap::new();

    // Run formatting
    let mut in_rpr = false;
    let mut run_fmt = RunFmt::default();

    // Per-run segments for the current paragraph (Phase C: mixed formatting)
    let mut para_runs: Vec<RunSegment> = Vec::new();
    let mut current_run_text = String::new();

    // Hyperlink tracking
    let mut in_hyperlink = false;
    let mut hyperlink_url: Option<String> = None;

    // Drawing/image tracking
    let mut in_drawing = false;
    let mut drawing_blip_rid: Option<String> = None;
    let mut drawing_alt_text: Option<String> = None;

    // OMML equation tracking
    let mut in_math = false;
    let mut math_depth: u32 = 0;
    let mut math_stack: Vec<MathFrame> = Vec::new();
    let mut math_is_display = false;

    // Pending images from this paragraph
    let mut pending_images: Vec<(Option<String>, Option<String>)> = Vec::new();

    // Comment tracking
    let mut active_comment_ids: Vec<String> = Vec::new();

    // VML image tracking (w:pict/v:imagedata)
    let mut in_pict = false;
    let mut pict_rid: Option<String> = None;

    // Phase A: suppress text from EMU attributes and field codes
    let mut suppress_text_depth: u32 = 0;
    let mut in_fld_char = false; // inside fldChar begin..end sequence

    // Phase B: mc:AlternateContent handling — only process mc:Choice, skip mc:Fallback
    let mut skip_fallback_depth: u32 = 0;

    // Phase G: textbox tracking
    let mut in_txbx_content = false;
    let mut txbx_texts: Vec<String> = Vec::new();
    let mut txbx_para_text = String::new();
    let mut txbx_in_run = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let local = get_local_name(e);

                // Phase B: skip mc:Fallback entirely
                if skip_fallback_depth > 0 {
                    skip_fallback_depth += 1;
                    continue;
                }
                if local == "Fallback" {
                    skip_fallback_depth = 1;
                    continue;
                }

                // Phase A: suppress EMU/field text
                if is_suppressed_text_element(&local) {
                    suppress_text_depth += 1;
                    continue;
                }

                match local.as_str() {
                    "tbl" => {
                        table_stack.push(TableState::new());
                    }
                    "tr" if !table_stack.is_empty() => {
                        if let Some(ts) = table_stack.last_mut() {
                            ts.col = 0;
                        }
                    }
                    "tc" if !table_stack.is_empty() => {
                        if let Some(ts) = table_stack.last_mut() {
                            ts.in_cell = true;
                            ts.cell_text.clear();
                            ts.cell_content = RichCellContent::new();
                            ts.current_col_span = 1;
                            ts.prev_para_was_list = false;
                            ts.vmerge_continues = false;
                        }
                    }
                    "tcPr" if !table_stack.is_empty() => {
                        if table_stack.last().is_some_and(|ts| ts.in_cell) {
                            in_tc_pr = true;
                        }
                    }
                    // Phase G: textbox content
                    "txbxContent" => {
                        in_txbx_content = true;
                        txbx_texts.clear();
                        txbx_para_text.clear();
                    }
                    "p" if in_txbx_content && !in_math => {
                        txbx_para_text.clear();
                    }
                    "r" if in_txbx_content && !in_math => {
                        txbx_in_run = true;
                    }
                    "p" if !in_math && !in_txbx_content => {
                        in_paragraph = true;
                        para_style_id.clear();
                        has_num_pr = false;
                        num_id.clear();
                        num_ilvl = 0;
                        para_runs.clear();
                        current_run_text.clear();
                        pending_images.clear();
                    }
                    "numPr" if in_paragraph => {
                        has_num_pr = true;
                    }
                    "hyperlink" if in_paragraph => {
                        in_hyperlink = true;
                        let rid = get_attr(e, "id");
                        hyperlink_url = rid.and_then(|id| {
                            let url = rels.resolve_url(&id);
                            if url.is_none() {
                                log::debug!("Hyperlink r:id '{}' not found in relationships", id);
                            }
                            url
                        });
                        if hyperlink_url.is_none() {
                            if let Some(anchor) = get_attr(e, "anchor") {
                                hyperlink_url = Some(format!("#{}", anchor));
                            }
                        }
                    }
                    "r" if in_paragraph && !in_math => {
                        in_run = true;
                        run_fmt = RunFmt::default();
                        current_run_text.clear();
                    }
                    "rPr" if in_run && !in_math => {
                        in_rpr = true;
                    }
                    "drawing" if in_paragraph || in_txbx_content => {
                        in_drawing = true;
                        drawing_blip_rid = None;
                        drawing_alt_text = None;
                    }
                    "pict" if in_paragraph || in_txbx_content => {
                        in_pict = true;
                        pict_rid = None;
                    }
                    "oMathPara" | "oMath" => {
                        if !in_math {
                            in_math = true;
                            math_depth = 1;
                            math_is_display = local == "oMathPara";
                            math_stack.clear();
                            math_stack.push(MathFrame::new(&local));
                        } else {
                            math_depth += 1;
                            if is_math_structural(&local) || is_math_slot(&local) {
                                math_stack.push(MathFrame::new(&local));
                            }
                        }
                    }
                    _ if in_math => {
                        math_depth += 1;
                        if is_math_structural(&local) || is_math_slot(&local) {
                            math_stack.push(MathFrame::new(&local));
                        }
                    }
                    "commentRangeStart" => {
                        if let Some(id) = get_attr(e, "id") {
                            active_comment_ids.push(id);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let local = get_local_name(e);

                if skip_fallback_depth > 0 {
                    continue;
                }

                match local.as_str() {
                    "p" if !in_math && !in_txbx_content => {
                        let in_table_cell = table_stack.last().is_some_and(|ts| ts.in_cell);
                        if !in_table_cell {
                            list_nesting.reset();
                            doc.add_text(DocItemLabel::Text, "", current_parent.as_deref());
                        }
                    }
                    "pStyle" if in_paragraph && !in_math => {
                        para_style_id = get_attr(e, "val").unwrap_or_default();
                    }
                    "numId" if has_num_pr => {
                        num_id = get_attr(e, "val").unwrap_or_default();
                    }
                    "ilvl" if has_num_pr => {
                        num_ilvl = get_attr(e, "val").and_then(|v| v.parse().ok()).unwrap_or(0);
                    }
                    "gridSpan" if in_tc_pr => {
                        if let Some(ts) = table_stack.last_mut() {
                            ts.current_col_span =
                                get_attr(e, "val").and_then(|v| v.parse().ok()).unwrap_or(1);
                        }
                    }
                    "vMerge" if in_tc_pr => {
                        if let Some(ts) = table_stack.last_mut() {
                            let val = get_attr(e, "val").unwrap_or_default();
                            if val != "restart" {
                                ts.vmerge_continues = true;
                            }
                        }
                    }
                    "b" if in_rpr => run_fmt.bold = is_toggle_on(e),
                    "i" if in_rpr => run_fmt.italic = is_toggle_on(e),
                    "u" if in_rpr => run_fmt.underline = is_toggle_on(e),
                    "strike" if in_rpr => run_fmt.strikethrough = is_toggle_on(e),
                    "vertAlign" if in_rpr => {
                        run_fmt.script = get_attr(e, "val");
                    }
                    "br" if in_run && !in_math => {
                        current_run_text.push('\n');
                    }
                    "blip" if in_drawing => {
                        drawing_blip_rid = get_attr(e, "embed").or_else(|| get_attr(e, "link"));
                    }
                    "docPr" if in_drawing => {
                        drawing_alt_text = get_attr(e, "descr");
                    }
                    "imagedata" if in_pict => {
                        pict_rid = get_attr(e, "id");
                    }
                    "fldChar" => {
                        let fld_type = get_attr(e, "fldCharType").unwrap_or_default();
                        match fld_type.as_str() {
                            "begin" => in_fld_char = true,
                            "end" => in_fld_char = false,
                            "separate" => {} // between instrText and result
                            _ => {}
                        }
                    }
                    "commentRangeStart" => {
                        if let Some(id) = get_attr(e, "id") {
                            active_comment_ids.push(id);
                        }
                    }
                    "commentRangeEnd" => {
                        if let Some(id) = get_attr(e, "id") {
                            if let Some(text) = comments.get(&id) {
                                let in_table_cell = table_stack.last().is_some_and(|ts| ts.in_cell);
                                if !in_table_cell {
                                    doc.add_text(
                                        DocItemLabel::Footnote,
                                        text,
                                        current_parent.as_deref(),
                                    );
                                }
                            }
                            active_comment_ids.retain(|c| c != &id);
                        }
                    }
                    _ if in_math => {
                        handle_math_empty(e, &local, &mut math_stack);
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if skip_fallback_depth > 0 || suppress_text_depth > 0 {
                    continue;
                }

                if in_math {
                    let raw = String::from_utf8_lossy(e.as_ref()).to_string();
                    let t = unicode_math_to_latex(&raw);
                    if let Some(frame) = math_stack.last_mut() {
                        if t.starts_with('\\')
                            && !frame.text.is_empty()
                            && !frame.text.ends_with(' ')
                            && !frame.text.ends_with('{')
                        {
                            frame.text.push(' ');
                        }
                        frame.text.push_str(&t);
                    }
                } else if in_txbx_content && txbx_in_run {
                    let t = String::from_utf8_lossy(e.as_ref()).to_string();
                    txbx_para_text.push_str(&t);
                } else if in_run && in_paragraph {
                    let t = String::from_utf8_lossy(e.as_ref()).to_string();

                    // Phase A: suppress field instruction text
                    if in_fld_char && is_field_code_text(&t) {
                        continue;
                    }

                    current_run_text.push_str(&t);
                }
            }
            Ok(Event::End(ref e)) => {
                let local = get_local_name_end(e);

                // Phase B: track Fallback end
                if skip_fallback_depth > 0 {
                    skip_fallback_depth -= 1;
                    continue;
                }

                // Phase A: suppress element end
                if suppress_text_depth > 0 {
                    if is_suppressed_text_element(&local) {
                        suppress_text_depth -= 1;
                    }
                    continue;
                }

                match local.as_str() {
                    "rPr" if !in_math => {
                        in_rpr = false;
                    }
                    "r" if !in_math && in_txbx_content => {
                        txbx_in_run = false;
                    }
                    "r" if !in_math => {
                        if in_run && !current_run_text.is_empty() {
                            let link = if in_hyperlink {
                                hyperlink_url.clone()
                            } else {
                                None
                            };
                            // Merge consecutive runs with same hyperlink AND same formatting
                            let merged = if let Some(last) = para_runs.last_mut() {
                                if last.hyperlink == link && last.fmt == run_fmt {
                                    last.text.push_str(&current_run_text);
                                    true
                                } else {
                                    false
                                }
                            } else {
                                false
                            };
                            if !merged {
                                para_runs.push(RunSegment {
                                    text: current_run_text.clone(),
                                    fmt: run_fmt.clone(),
                                    hyperlink: link,
                                    is_formula: false,
                                });
                            }
                            in_run = false;
                            current_run_text.clear();
                        } else {
                            in_run = false;
                            current_run_text.clear();
                        }
                    }
                    "hyperlink" => {
                        in_hyperlink = false;
                        hyperlink_url = None;
                    }
                    "drawing" => {
                        if in_drawing {
                            if drawing_blip_rid.is_some() {
                                pending_images
                                    .push((drawing_blip_rid.take(), drawing_alt_text.take()));
                            } else {
                                // DrawingML shape without blip — emit placeholder
                                pending_images.push((None, drawing_alt_text.take()));
                            }
                            in_drawing = false;
                        }
                    }
                    "pict" => {
                        if in_pict {
                            if let Some(ref rid) = pict_rid {
                                pending_images.push((Some(rid.clone()), None));
                            }
                            in_pict = false;
                        }
                    }
                    "tcPr" => in_tc_pr = false,
                    // Phase G: textbox paragraph end
                    "p" if in_txbx_content && !in_math => {
                        let t = txbx_para_text.trim().to_string();
                        if !t.is_empty() {
                            txbx_texts.push(t);
                        }
                        txbx_para_text.clear();
                    }
                    // Phase G: textbox content end — emit all text
                    "txbxContent" => {
                        in_txbx_content = false;
                        for txt in txbx_texts.drain(..) {
                            let in_table_cell = table_stack.last().is_some_and(|ts| ts.in_cell);
                            if in_table_cell {
                                if let Some(ts) = table_stack.last_mut() {
                                    if !ts.cell_text.is_empty() {
                                        ts.cell_text.push('\n');
                                    }
                                    ts.cell_text.push_str(&txt);
                                }
                            } else {
                                doc.add_text(DocItemLabel::Text, &txt, current_parent.as_deref());
                            }
                        }
                    }
                    "p" if !in_math && !in_txbx_content => {
                        in_paragraph = false;
                        let in_table_cell = table_stack.last().is_some_and(|ts| ts.in_cell);
                        if in_table_cell {
                            let combined: String =
                                para_runs.iter().map(|r| r.text.as_str()).collect();
                            let is_list_para = has_num_pr && !num_id.is_empty() && num_id != "0";
                            if let Some(ts) = table_stack.last_mut() {
                                if !ts.cell_text.is_empty() && !combined.is_empty() {
                                    if is_list_para || ts.prev_para_was_list {
                                        ts.cell_text.push('\n');
                                    } else {
                                        ts.cell_text.push_str("\n\n");
                                    }
                                }
                                ts.cell_text.push_str(&combined);
                                ts.prev_para_was_list = is_list_para;
                                ts.cell_content.paragraphs.push(CellParagraph {
                                    runs: para_runs.clone(),
                                    has_num_pr,
                                    num_id: num_id.clone(),
                                    num_ilvl,
                                    style_id: para_style_id.clone(),
                                });
                                for img in pending_images.drain(..) {
                                    ts.cell_content.images.push(img);
                                }
                            }
                        } else {
                            let had_images = !pending_images.is_empty();
                            // Emit pending images first
                            for (img_rid, alt) in pending_images.drain(..) {
                                if let Some(rid) = img_rid {
                                    emit_image(
                                        &rid,
                                        alt.as_deref(),
                                        rels,
                                        media,
                                        doc,
                                        current_parent.as_deref(),
                                    );
                                } else {
                                    doc.add_picture(alt.as_deref(), current_parent.as_deref());
                                }
                            }

                            let full_text: String =
                                para_runs.iter().map(|r| r.text.as_str()).collect();
                            let text = full_text.trim();

                            if !text.is_empty() {
                                let is_title_style =
                                    styles.title_style_ids.contains(&para_style_id);
                                if is_title_style {
                                    let idx = doc.add_title(text, None);
                                    current_parent = Some(format!("#/texts/{}", idx));
                                    list_nesting.reset();
                                } else if let Some(level) =
                                    styles.resolve_heading_level(&para_style_id)
                                {
                                    // Resolve numbering: explicit numPr or style-level numPr
                                    let effective_num =
                                        if has_num_pr && !num_id.is_empty() && num_id != "0" {
                                            Some((num_id.clone(), num_ilvl))
                                        } else {
                                            styles.style_num_pr.get(&para_style_id).cloned()
                                        };

                                    let heading_text =
                                        if let Some((eff_num_id, eff_ilvl)) = effective_num {
                                            let counter = heading_counters
                                                .entry((eff_num_id.clone(), eff_ilvl))
                                                .or_insert(0);
                                            *counter += 1;

                                            let keys_to_reset: Vec<_> = heading_counters
                                                .keys()
                                                .filter(|(nid, lvl)| {
                                                    nid == &eff_num_id && *lvl > eff_ilvl
                                                })
                                                .cloned()
                                                .collect();
                                            for key in keys_to_reset {
                                                heading_counters.insert(key, 0);
                                            }

                                            let mut prefix_parts = Vec::new();
                                            for lvl in 0..=eff_ilvl {
                                                let c = heading_counters
                                                    .entry((eff_num_id.clone(), lvl))
                                                    .or_insert(0);
                                                if *c == 0 {
                                                    *c = 1;
                                                }
                                                prefix_parts.push(c.to_string());
                                            }
                                            let prefix = prefix_parts.join(".");
                                            if prefix.is_empty() {
                                                text.to_string()
                                            } else {
                                                format!("{} {}", prefix, text)
                                            }
                                        } else {
                                            text.to_string()
                                        };
                                    let idx = doc.add_section_header(&heading_text, level, None);
                                    current_parent = Some(format!("#/texts/{}", idx));
                                    list_nesting.reset();
                                } else if has_num_pr && num_id != "0" && !num_id.is_empty() {
                                    let (is_ordered, _) = numbering
                                        .definitions
                                        .get(&(num_id.clone(), num_ilvl))
                                        .map(|(ordered, _fmt)| (*ordered, ""))
                                        .unwrap_or((false, ""));

                                    let gref = list_nesting.get_or_create_group(
                                        &num_id,
                                        num_ilvl,
                                        is_ordered,
                                        doc,
                                        current_parent.as_deref(),
                                    );
                                    let marker = list_nesting.next_marker(is_ordered);

                                    let has_mixed = para_runs.len() >= 2 && {
                                        let first_fmt = &para_runs[0].fmt;
                                        para_runs
                                            .iter()
                                            .skip(1)
                                            .any(|r| r.fmt != *first_fmt || r.hyperlink.is_some())
                                    };

                                    if has_mixed {
                                        let li_idx =
                                            doc.add_list_item("", is_ordered, Some(&marker), &gref);
                                        let li_ref = format!("#/texts/{}", li_idx);
                                        let inline_idx = doc.add_group(
                                            "group",
                                            GroupLabel::Inline,
                                            Some(&li_ref),
                                        );
                                        let inline_ref = format!("#/groups/{}", inline_idx);
                                        for run in &para_runs {
                                            let run_text = run.text.trim();
                                            if run_text.is_empty() {
                                                continue;
                                            }
                                            doc.add_text_ext(
                                                DocItemLabel::Text,
                                                run_text,
                                                Some(&inline_ref),
                                                run.fmt.to_text_formatting(),
                                                run.hyperlink.clone(),
                                            );
                                        }
                                    } else {
                                        doc.add_list_item(text, is_ordered, Some(&marker), &gref);
                                    }
                                } else {
                                    list_nesting.reset();
                                    // Phase C: emit per-run formatting
                                    emit_paragraph_runs(&para_runs, doc, current_parent.as_deref());
                                }
                            } else {
                                for (img_rid, alt) in pending_images.drain(..) {
                                    if let Some(rid) = img_rid {
                                        emit_image(
                                            &rid,
                                            alt.as_deref(),
                                            rels,
                                            media,
                                            doc,
                                            current_parent.as_deref(),
                                        );
                                    } else {
                                        doc.add_picture(alt.as_deref(), current_parent.as_deref());
                                    }
                                }
                                if !had_images {
                                    if has_num_pr && num_id != "0" && !num_id.is_empty() {
                                        let (is_ordered, _) = numbering
                                            .definitions
                                            .get(&(num_id.clone(), num_ilvl))
                                            .map(|(ordered, _fmt)| (*ordered, ""))
                                            .unwrap_or((false, ""));
                                        let gref = list_nesting.get_or_create_group(
                                            &num_id,
                                            num_ilvl,
                                            is_ordered,
                                            doc,
                                            current_parent.as_deref(),
                                        );
                                        let marker = list_nesting.next_marker(is_ordered);
                                        doc.add_list_item("", is_ordered, Some(&marker), &gref);
                                    } else if styles.resolve_heading_level(&para_style_id).is_none()
                                    {
                                        list_nesting.reset();
                                        doc.add_text(
                                            DocItemLabel::Text,
                                            "",
                                            current_parent.as_deref(),
                                        );
                                    }
                                }
                            }
                        }
                        para_runs.clear();
                    }
                    "tc" if !table_stack.is_empty() => {
                        if let Some(ts) = table_stack.last_mut() {
                            ts.in_cell = false;
                            if ts.vmerge_continues {
                                let row = ts.row;
                                let col = ts.col;
                                ts.vmerge_continuations.push((row, col));
                                ts.col += ts.current_col_span;
                            } else {
                                let text = ts.cell_text.clone();
                                let col_span = ts.current_col_span;
                                let row = ts.row;
                                let col = ts.col;
                                let rich =
                                    std::mem::replace(&mut ts.cell_content, RichCellContent::new());
                                let md = rich.to_markdown();
                                let formatted =
                                    if !md.is_empty() && md.len() >= text.len() && md != text {
                                        Some(md)
                                    } else {
                                        None
                                    };
                                ts.rich_cells.push((row, col, col_span, rich));
                                ts.cells.push(TableCell {
                                    row_span: 1,
                                    col_span,
                                    start_row_offset_idx: row,
                                    end_row_offset_idx: row + 1,
                                    start_col_offset_idx: col,
                                    end_col_offset_idx: col + col_span,
                                    text,
                                    column_header: row == 0,
                                    row_header: false,
                                    row_section: false,
                                    fillable: false,
                                    formatted_text: formatted,
                                });
                                ts.col += col_span;
                            }
                        }
                    }
                    "tr" if !table_stack.is_empty() => {
                        if let Some(ts) = table_stack.last_mut() {
                            ts.max_cols = ts.max_cols.max(ts.col);
                            ts.row += 1;
                        }
                    }
                    "tbl" if !table_stack.is_empty() => {
                        let mut ts = table_stack.pop().unwrap();
                        patch_vmerge_row_spans(&mut ts);

                        // Phase D: single-cell table → treat as body content
                        if ts.row == 1 && ts.max_cols == 1 && ts.cells.len() == 1 {
                            if let Some((_, _, _, ref rich)) = ts.rich_cells.first() {
                                let mut sc_list_nesting = ListNesting::new();
                                for para in &rich.paragraphs {
                                    let full: String =
                                        para.runs.iter().map(|r| r.text.as_str()).collect();
                                    let text = full.trim();
                                    if para.has_num_pr
                                        && !para.num_id.is_empty()
                                        && para.num_id != "0"
                                    {
                                        let (is_ordered, _) = numbering
                                            .definitions
                                            .get(&(para.num_id.clone(), para.num_ilvl))
                                            .map(|(ordered, _fmt)| (*ordered, ""))
                                            .unwrap_or((false, ""));
                                        let gref = sc_list_nesting.get_or_create_group(
                                            &para.num_id,
                                            para.num_ilvl,
                                            is_ordered,
                                            doc,
                                            current_parent.as_deref(),
                                        );
                                        let marker = sc_list_nesting.next_marker(is_ordered);
                                        let sc_has_mixed = para.runs.len() >= 2 && {
                                            let first_fmt = &para.runs[0].fmt;
                                            para.runs.iter().skip(1).any(|r| {
                                                r.fmt != *first_fmt || r.hyperlink.is_some()
                                            })
                                        };
                                        if sc_has_mixed {
                                            let li_idx = doc.add_list_item(
                                                "",
                                                is_ordered,
                                                Some(&marker),
                                                &gref,
                                            );
                                            let li_ref = format!("#/texts/{}", li_idx);
                                            let inline_idx = doc.add_group(
                                                "group",
                                                GroupLabel::Inline,
                                                Some(&li_ref),
                                            );
                                            let inline_ref = format!("#/groups/{}", inline_idx);
                                            for run in &para.runs {
                                                let run_text = run.text.trim();
                                                if run_text.is_empty() {
                                                    continue;
                                                }
                                                doc.add_text_ext(
                                                    DocItemLabel::Text,
                                                    run_text,
                                                    Some(&inline_ref),
                                                    run.fmt.to_text_formatting(),
                                                    run.hyperlink.clone(),
                                                );
                                            }
                                        } else {
                                            doc.add_list_item(
                                                text,
                                                is_ordered,
                                                Some(&marker),
                                                &gref,
                                            );
                                        }
                                    } else {
                                        sc_list_nesting.reset();
                                        if !text.is_empty() {
                                            emit_paragraph_runs(
                                                &para.runs,
                                                doc,
                                                current_parent.as_deref(),
                                            );
                                        } else {
                                            doc.add_text(
                                                DocItemLabel::Text,
                                                "",
                                                current_parent.as_deref(),
                                            );
                                        }
                                    }
                                }
                                for (img_rid, alt) in &rich.images {
                                    if let Some(rid) = img_rid {
                                        emit_image(
                                            rid,
                                            alt.as_deref(),
                                            rels,
                                            media,
                                            doc,
                                            current_parent.as_deref(),
                                        );
                                    } else {
                                        doc.add_picture(alt.as_deref(), current_parent.as_deref());
                                    }
                                }
                            } else {
                                let cell_text = ts.cells[0].text.trim().to_string();
                                if !cell_text.is_empty() {
                                    for line in cell_text.split('\n') {
                                        let line = line.trim();
                                        if !line.is_empty() {
                                            doc.add_text(
                                                DocItemLabel::Text,
                                                line,
                                                current_parent.as_deref(),
                                            );
                                        }
                                    }
                                }
                            }
                        } else if !ts.cells.is_empty() {
                            let table_idx = doc.tables_len();
                            doc.add_table(ts.cells, ts.row, ts.max_cols, current_parent.as_deref());
                            let table_ref = format!("#/tables/{}", table_idx);
                            emit_rich_cell_groups(
                                &ts.rich_cells,
                                table_idx,
                                &table_ref,
                                numbering,
                                styles,
                                rels,
                                media,
                                doc,
                            );
                        }
                    }
                    // commentRangeEnd is handled as Event::Empty (self-closing tag)
                    // OMML equation end
                    "oMathPara" | "oMath" if in_math => {
                        math_depth -= 1;
                        if is_math_structural(&local) || is_math_slot(&local) {
                            pop_math_frame(&local, &mut math_stack);
                        }
                        if math_depth == 0 {
                            in_math = false;
                            let latex = math_stack.pop().map(|f| f.text).unwrap_or_default();
                            let latex = latex.trim().to_string();
                            if !latex.is_empty() {
                                let in_table_cell = table_stack.last().is_some_and(|ts| ts.in_cell);
                                if in_table_cell {
                                    if let Some(ts) = table_stack.last_mut() {
                                        if !ts.cell_text.is_empty() {
                                            ts.cell_text.push(' ');
                                        }
                                        // Phase F: inline eq in table cells gets $...$
                                        ts.cell_text.push('$');
                                        ts.cell_text.push_str(&latex);
                                        ts.cell_text.push('$');
                                    }
                                } else if math_is_display {
                                    doc.add_text(
                                        DocItemLabel::Formula,
                                        &latex,
                                        current_parent.as_deref(),
                                    );
                                } else {
                                    para_runs.push(RunSegment {
                                        text: latex.clone(),
                                        fmt: RunFmt::default(),
                                        hyperlink: None,
                                        is_formula: true,
                                    });
                                }
                            }
                            math_stack.clear();
                        }
                    }
                    _ if in_math => {
                        math_depth -= 1;
                        if is_math_structural(&local) || is_math_slot(&local) {
                            pop_math_frame(&local, &mut math_stack);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::GeneralRef(ref e)) => {
                let entity = String::from_utf8_lossy(e.as_ref()).to_string();
                let ch = match entity.as_str() {
                    "amp" => "&",
                    "lt" => "<",
                    "gt" => ">",
                    "apos" => "'",
                    "quot" => "\"",
                    _ => "",
                };
                if !ch.is_empty() && (skip_fallback_depth == 0 && suppress_text_depth == 0) {
                    if in_math {
                        if let Some(frame) = math_stack.last_mut() {
                            frame.text.push_str(ch);
                        }
                    } else if in_txbx_content && txbx_in_run {
                        txbx_para_text.push_str(ch);
                    } else if in_run && in_paragraph {
                        current_run_text.push_str(ch);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(anyhow::anyhow!("XML parse error in document.xml: {}", e));
            }
            _ => {}
        }
    }

    // Emit any collected comments
    for (_id, text) in comments {
        if !text.is_empty() && active_comment_ids.iter().any(|c| c == _id) {
            doc.add_text(DocItemLabel::Footnote, text, None);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase C: Emit paragraph with per-run formatting
// ---------------------------------------------------------------------------

fn emit_paragraph_runs(runs: &[RunSegment], doc: &mut DoclingDocument, parent_ref: Option<&str>) {
    if runs.is_empty() {
        return;
    }

    let has_formula = runs.iter().any(|r| r.is_formula);
    let all_same = !has_formula
        && runs
            .iter()
            .all(|r| r.fmt == runs[0].fmt && r.hyperlink.is_none());

    if all_same {
        let full_text: String = runs.iter().map(|r| r.text.as_str()).collect();
        let text = full_text.trim();
        if !text.is_empty() {
            doc.add_text_ext(
                DocItemLabel::Text,
                text,
                parent_ref,
                runs[0].fmt.to_text_formatting(),
                None,
            );
        }
        return;
    }

    // Merge consecutive runs with the same formatting before emitting
    let mut merged: Vec<RunSegment> = Vec::new();
    for run in runs {
        if let Some(last) = merged.last_mut() {
            if last.fmt == run.fmt
                && last.hyperlink == run.hyperlink
                && last.is_formula == run.is_formula
            {
                last.text.push_str(&run.text);
                continue;
            }
        }
        merged.push(run.clone());
    }

    let inline_idx = doc.add_group("group", GroupLabel::Inline, parent_ref);
    let inline_ref = format!("#/groups/{}", inline_idx);

    for run in &merged {
        let text = run.text.trim();
        if text.is_empty() {
            continue;
        }
        let label = if run.is_formula {
            DocItemLabel::Formula
        } else {
            DocItemLabel::Text
        };
        doc.add_text_ext(
            label,
            text,
            Some(&inline_ref),
            run.fmt.to_text_formatting(),
            run.hyperlink.clone(),
        );
    }
}

// ---------------------------------------------------------------------------
// Rich cell group emission
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_rich_cell_groups(
    rich_cells: &[(u32, u32, u32, RichCellContent)],
    table_idx: usize,
    table_ref: &str,
    numbering: &NumberingInfo,
    styles: &Styles,
    rels: &Relationships,
    media: &HashMap<String, Vec<u8>>,
    doc: &mut DoclingDocument,
) {
    let any_rich = rich_cells.iter().any(|(_, _, _, content)| {
        let has_images = !content.images.is_empty();
        let has_lists = content
            .paragraphs
            .iter()
            .any(|p| p.has_num_pr && !p.num_id.is_empty() && p.num_id != "0");
        let has_mixed_fmt = content.paragraphs.iter().any(|p| {
            if p.runs.len() < 2 {
                return false;
            }
            let first_fmt = &p.runs[0].fmt;
            p.runs.iter().skip(1).any(|r| r.fmt != *first_fmt)
        });
        let non_empty_para_count = content
            .paragraphs
            .iter()
            .filter(|p| p.runs.iter().any(|r| !r.text.trim().is_empty()))
            .count();
        let multi_para = non_empty_para_count > 1;
        has_images || has_lists || has_mixed_fmt || multi_para
    });

    if !any_rich {
        return;
    }

    for (row, col, _col_span, content) in rich_cells {
        let has_any_content =
            !content.images.is_empty() || content.paragraphs.iter().any(|p| !p.runs.is_empty());
        if !has_any_content {
            continue;
        }

        let group_name = format!("rich_cell_group_{}_{}_{}", table_idx + 1, row, col);

        let gidx = doc.add_group(&group_name, GroupLabel::Unspecified, Some(table_ref));
        let group_ref = format!("#/groups/{}", gidx);

        for (img_rid, alt) in &content.images {
            if let Some(rid) = img_rid {
                emit_image(rid, alt.as_deref(), rels, media, doc, Some(&group_ref));
            } else {
                doc.add_picture(alt.as_deref(), Some(&group_ref));
            }
        }

        let mut cell_list_nesting = ListNesting::new();

        for para in &content.paragraphs {
            let all_same = para.runs.is_empty()
                || para
                    .runs
                    .iter()
                    .all(|r| r.fmt == para.runs[0].fmt && r.hyperlink.is_none());
            let full_text: String = para.runs.iter().map(|r| r.text.as_str()).collect();
            let text = full_text.trim();

            if para.has_num_pr && !para.num_id.is_empty() && para.num_id != "0" {
                let (is_ordered, _) = numbering
                    .definitions
                    .get(&(para.num_id.clone(), para.num_ilvl))
                    .map(|(ordered, _)| (*ordered, ""))
                    .unwrap_or((false, ""));
                let gref = cell_list_nesting.get_or_create_group(
                    &para.num_id,
                    para.num_ilvl,
                    is_ordered,
                    doc,
                    Some(&group_ref),
                );
                let marker = cell_list_nesting.next_marker(is_ordered);
                let cell_li_mixed = para.runs.len() >= 2 && {
                    let first_fmt = &para.runs[0].fmt;
                    para.runs
                        .iter()
                        .skip(1)
                        .any(|r| r.fmt != *first_fmt || r.hyperlink.is_some())
                };
                if cell_li_mixed {
                    let li_idx = doc.add_list_item("", is_ordered, Some(&marker), &gref);
                    let li_ref = format!("#/texts/{}", li_idx);
                    let inline_idx = doc.add_group("group", GroupLabel::Inline, Some(&li_ref));
                    let inline_ref = format!("#/groups/{}", inline_idx);
                    for run in &para.runs {
                        let run_text = run.text.trim();
                        if run_text.is_empty() {
                            continue;
                        }
                        doc.add_text_ext(
                            DocItemLabel::Text,
                            run_text,
                            Some(&inline_ref),
                            run.fmt.to_text_formatting(),
                            run.hyperlink.clone(),
                        );
                    }
                } else {
                    doc.add_list_item(text, is_ordered, Some(&marker), &gref);
                }
            } else {
                cell_list_nesting.reset();

                if text.is_empty() && para.runs.is_empty() {
                    doc.add_text(DocItemLabel::Text, "", Some(&group_ref));
                } else if all_same {
                    if !text.is_empty() {
                        let is_title = styles.title_style_ids.contains(&para.style_id);
                        if is_title {
                            doc.add_title(text, Some(&group_ref));
                        } else if let Some(level) = styles.resolve_heading_level(&para.style_id) {
                            doc.add_section_header(text, level, Some(&group_ref));
                        } else {
                            doc.add_text_ext(
                                DocItemLabel::Text,
                                text,
                                Some(&group_ref),
                                para.runs.first().and_then(|r| r.fmt.to_text_formatting()),
                                None,
                            );
                        }
                    } else {
                        doc.add_text(DocItemLabel::Text, "", Some(&group_ref));
                    }
                } else {
                    let inline_idx = doc.add_group("group", GroupLabel::Inline, Some(&group_ref));
                    let inline_ref = format!("#/groups/{}", inline_idx);
                    for run in &para.runs {
                        let run_text = run.text.trim();
                        if run_text.is_empty() {
                            continue;
                        }
                        doc.add_text_ext(
                            DocItemLabel::Text,
                            run_text,
                            Some(&inline_ref),
                            run.fmt.to_text_formatting(),
                            run.hyperlink.clone(),
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Image emission
// ---------------------------------------------------------------------------

fn emit_image(
    rid: &str,
    alt_text: Option<&str>,
    rels: &Relationships,
    media: &HashMap<String, Vec<u8>>,
    doc: &mut DoclingDocument,
    parent_ref: Option<&str>,
) {
    let media_path = match rels.resolve_media_path(rid) {
        Some(p) => p,
        None => {
            log::debug!("Image r:id '{}' not found in relationships", rid);
            return;
        }
    };
    let data = match media.get(&media_path) {
        Some(d) => d,
        None => {
            log::debug!("Media file '{}' not found in archive", media_path);
            return;
        }
    };

    let idx = doc.add_picture(alt_text, parent_ref);

    let ext = media_path
        .rsplit('.')
        .next()
        .unwrap_or("png")
        .to_lowercase();
    let mimetype = image_mimetype(&ext);

    let (width, height) = match image::load_from_memory(data) {
        Ok(img) => (img.width() as f64, img.height() as f64),
        Err(_) => (0.0, 0.0),
    };

    let b64 = base64::engine::general_purpose::STANDARD.encode(data);
    let uri = format!("data:{};base64,{}", mimetype, b64);

    doc.set_picture_image(
        idx,
        ImageRef {
            mimetype: mimetype.to_string(),
            dpi: 72,
            size: ImageSize { width, height },
            uri,
        },
    );
}

fn image_mimetype(ext: &str) -> &'static str {
    match ext {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        "webp" => "image/webp",
        "emf" => "image/x-emf",
        "wmf" => "image/x-wmf",
        "svg" => "image/svg+xml",
        _ => "image/png",
    }
}

// ---------------------------------------------------------------------------
// vMerge patching (Phase H: propagate text to continuation rows)
// ---------------------------------------------------------------------------

fn patch_vmerge_row_spans(ts: &mut TableState) {
    for &(cont_row, cont_col) in &ts.vmerge_continuations {
        let mut best_idx: Option<usize> = None;
        let mut best_row: u32 = 0;
        for (idx, cell) in ts.cells.iter().enumerate() {
            if cell.start_col_offset_idx == cont_col
                && cell.start_row_offset_idx < cont_row
                && (best_idx.is_none() || cell.start_row_offset_idx > best_row)
            {
                best_idx = Some(idx);
                best_row = cell.start_row_offset_idx;
            }
        }
        if let Some(idx) = best_idx {
            ts.cells[idx].row_span += 1;
            ts.cells[idx].end_row_offset_idx = cont_row + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// OMML to LaTeX converter
// ---------------------------------------------------------------------------

struct MathFrame {
    tag: String,
    text: String,
    named_parts: HashMap<String, String>,
    chr: Option<String>,
    beg_chr: Option<String>,
    end_chr: Option<String>,
}

impl MathFrame {
    fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            text: String::new(),
            named_parts: HashMap::new(),
            chr: None,
            beg_chr: None,
            end_chr: None,
        }
    }
}

fn is_math_structural(name: &str) -> bool {
    matches!(
        name,
        "f" | "sSup"
            | "sSub"
            | "sSubSup"
            | "rad"
            | "nary"
            | "d"
            | "acc"
            | "bar"
            | "m"
            | "eqArr"
            | "limLow"
            | "limUpp"
            | "func"
            | "groupChr"
            | "borderBox"
            | "box"
            | "sPre"
            | "oMath"
            | "oMathPara"
    )
}

fn is_math_slot(name: &str) -> bool {
    matches!(
        name,
        "num" | "den" | "e" | "sub" | "sup" | "deg" | "lim" | "fName" | "mr"
    )
}

fn handle_math_empty(e: &quick_xml::events::BytesStart, local: &str, stack: &mut [MathFrame]) {
    match local {
        "chr" => {
            if let Some(frame) = stack.last_mut() {
                if let Some(val) = get_attr(e, "val") {
                    frame.chr = Some(val);
                }
            }
        }
        "begChr" => {
            if let Some(frame) = stack.last_mut() {
                if let Some(val) = get_attr(e, "val") {
                    frame.beg_chr = Some(val);
                }
            }
        }
        "endChr" => {
            if let Some(frame) = stack.last_mut() {
                if let Some(val) = get_attr(e, "val") {
                    frame.end_chr = Some(val);
                }
            }
        }
        _ => {}
    }
}

fn pop_math_frame(local: &str, stack: &mut Vec<MathFrame>) {
    if stack.len() <= 1 {
        return;
    }

    if is_math_slot(local) {
        if let Some(frame) = stack.pop() {
            if let Some(parent) = stack.last_mut() {
                parent.named_parts.insert(frame.tag.clone(), frame.text);
            }
        }
    } else if is_math_structural(local) {
        if let Some(frame) = stack.pop() {
            let latex = format_math_element(&frame);
            if let Some(parent) = stack.last_mut() {
                parent.text.push_str(&latex);
            }
        }
    }
}

fn format_math_element(frame: &MathFrame) -> String {
    let get_part = |name: &str| -> &str {
        frame
            .named_parts
            .get(name)
            .map(|s| s.as_str())
            .unwrap_or("")
    };

    match frame.tag.as_str() {
        "f" => {
            format!("\\frac{{{}}}{{{}}}", get_part("num"), get_part("den"))
        }
        "sSup" => {
            format!("{}^{{{}}}", get_part("e"), get_part("sup"))
        }
        "sSub" => {
            format!("{}_{{{}}}", get_part("e"), get_part("sub"))
        }
        "sSubSup" => {
            format!(
                "{}_{{{}}}^{{{}}}",
                get_part("e"),
                get_part("sub"),
                get_part("sup")
            )
        }
        "sPre" => {
            let sub = get_part("sub");
            let sup = get_part("sup");
            let base = get_part("e");
            if !sub.is_empty() && !sup.is_empty() {
                format!("{{}}_{{{sub}}}^{{{sup}}}{base}")
            } else if !sub.is_empty() {
                format!("{{}}_{{{}}} {}", sub, base)
            } else if !sup.is_empty() {
                format!("{{}}^{{{}}} {}", sup, base)
            } else {
                base.to_string()
            }
        }
        "rad" => {
            let deg = get_part("deg");
            let base = get_part("e");
            if deg.is_empty() {
                format!("\\sqrt{{{}}}", base)
            } else {
                format!("\\sqrt[{}]{{{}}}", deg, base)
            }
        }
        "nary" => {
            let chr = frame.chr.as_deref().unwrap_or("∫");
            let cmd = nary_char_to_latex(chr);
            let sub = get_part("sub");
            let sup = get_part("sup");
            let base = get_part("e");
            let mut result = cmd.to_string();
            if !sub.is_empty() {
                result.push_str(&format!("_{{{}}}", sub));
            }
            if !sup.is_empty() {
                result.push_str(&format!("^{{{}}}", sup));
            }
            if !base.is_empty() {
                result.push(' ');
                result.push_str(base);
            }
            result
        }
        "d" => {
            let beg = frame.beg_chr.as_deref().unwrap_or("(");
            let end = frame.end_chr.as_deref().unwrap_or(")");
            let body = get_part("e");
            if body.is_empty() {
                format!("{}{}", &frame.text, "")
            } else {
                format!(
                    "\\left{}{}\\right{}",
                    escape_delim(beg),
                    body,
                    escape_delim(end)
                )
            }
        }
        "acc" => {
            let chr = frame.chr.as_deref().unwrap_or("\u{0302}");
            let cmd = accent_char_to_latex(chr);
            format!("{}{{{}}}", cmd, get_part("e"))
        }
        "bar" => {
            format!("\\overline{{{}}}", get_part("e"))
        }
        "limLow" => {
            let base = get_part("e");
            let lim = get_part("lim");
            format!("\\underset{{{}}}{{{}}}", lim, base)
        }
        "limUpp" => {
            let base = get_part("e");
            let lim = get_part("lim");
            format!("\\overset{{{}}}{{{}}}", lim, base)
        }
        "func" => {
            let fname = get_part("fName");
            let arg = get_part("e");
            format!("{} {}", fname, arg)
        }
        "groupChr" => get_part("e").to_string(),
        "borderBox" | "box" => get_part("e").to_string(),
        "eqArr" => {
            let parts: Vec<&str> = frame
                .named_parts
                .iter()
                .filter(|(k, _)| k.starts_with('e'))
                .map(|(_, v)| v.as_str())
                .collect();
            if parts.is_empty() {
                frame.text.clone()
            } else {
                parts.join(" \\\\ ")
            }
        }
        "m" => {
            let mut rows: Vec<String> = Vec::new();
            for (k, v) in &frame.named_parts {
                if k == "mr" {
                    rows.push(v.clone());
                }
            }
            if rows.is_empty() {
                frame.text.clone()
            } else {
                format!("\\begin{{matrix}} {} \\end{{matrix}}", rows.join(" \\\\ "))
            }
        }
        "mr" => {
            let parts: Vec<&str> = frame
                .named_parts
                .iter()
                .filter(|(k, _)| k.starts_with('e'))
                .map(|(_, v)| v.as_str())
                .collect();
            if parts.is_empty() {
                frame.text.clone()
            } else {
                parts.join(" & ")
            }
        }
        "oMath" | "oMathPara" => frame.text.clone(),
        _ => frame.text.clone(),
    }
}

// Phase F: add missing nary operators
fn nary_char_to_latex(chr: &str) -> &str {
    match chr {
        "∑" => "\\sum",
        "∏" => "\\prod",
        "∐" => "\\coprod",
        "∫" => "\\int",
        "∬" => "\\iint",
        "∭" => "\\iiint",
        "∮" => "\\oint",
        "∯" => "\\oiint",
        "∰" => "\\oiiint",
        "⋃" => "\\bigcup",
        "⋂" => "\\bigcap",
        "⋁" => "\\bigvee",
        "⋀" => "\\bigwedge",
        _ => "\\int",
    }
}

fn accent_char_to_latex(chr: &str) -> &str {
    match chr {
        "\u{0302}" | "^" => "\\hat",
        "\u{0303}" | "~" => "\\tilde",
        "\u{0307}" => "\\dot",
        "\u{0308}" => "\\ddot",
        "\u{20D7}" | "\u{2192}" => "\\vec",
        "\u{0304}" => "\\bar",
        "\u{0306}" => "\\breve",
        "\u{030C}" => "\\check",
        _ => "\\hat",
    }
}

fn unicode_math_to_latex(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for ch in text.chars() {
        let latex = match ch {
            'π' => "\\pi ",
            'α' => "\\alpha ",
            'β' => "\\beta ",
            'γ' => "\\gamma ",
            'δ' => "\\delta ",
            'ε' => "\\epsilon ",
            'θ' => "\\theta ",
            'λ' => "\\lambda ",
            'μ' => "\\mu ",
            'σ' => "\\sigma ",
            'φ' => "\\phi ",
            'ω' => "\\omega ",
            'Δ' => "\\Delta ",
            'Σ' => "\\Sigma ",
            'Ω' => "\\Omega ",
            '±' => "\\pm ",
            '∓' => "\\mp ",
            '×' => "\\times ",
            '÷' => "\\div ",
            '≤' => "\\leq ",
            '≥' => "\\geq ",
            '≠' => "\\neq ",
            '≈' => "\\approx ",
            '∞' => "\\infty ",
            '∈' => "\\in ",
            '∉' => "\\notin ",
            '⊂' => "\\subset ",
            '⊃' => "\\supset ",
            '∪' => "\\cup ",
            '∩' => "\\cap ",
            '→' => "\\rightarrow ",
            '←' => "\\leftarrow ",
            '⇒' => "\\Rightarrow ",
            '⇐' => "\\Leftarrow ",
            '∀' => "\\forall ",
            '∃' => "\\exists ",
            '∂' => "\\partial ",
            '∇' => "\\nabla ",
            _ => {
                result.push(ch);
                continue;
            }
        };
        if !result.is_empty() && !result.ends_with(' ') && !result.ends_with('{') {
            result.push(' ');
        }
        result.push_str(latex);
    }
    result
}

fn escape_delim(ch: &str) -> String {
    match ch {
        "{" => "\\{".to_string(),
        "}" => "\\}".to_string(),
        "|" => "|".to_string(),
        "‖" | "||" => "\\|".to_string(),
        "" => ".".to_string(),
        _ => ch.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Low-level XML helpers
// ---------------------------------------------------------------------------

fn read_zip_entry(
    archive: &mut zip::ZipArchive<std::fs::File>,
    name: &str,
) -> anyhow::Result<String> {
    let mut file = archive
        .by_name(name)
        .with_context(|| format!("Missing entry '{}' in ZIP", name))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

fn get_local_name(e: &quick_xml::events::BytesStart) -> String {
    let full = e.name().0;
    let s = std::str::from_utf8(full).unwrap_or("");
    s.rsplit(':').next().unwrap_or(s).to_string()
}

fn get_local_name_end(e: &quick_xml::events::BytesEnd) -> String {
    let full = e.name().0;
    let s = std::str::from_utf8(full).unwrap_or("");
    s.rsplit(':').next().unwrap_or(s).to_string()
}

fn is_toggle_on(e: &quick_xml::events::BytesStart) -> bool {
    match get_attr(e, "val") {
        None => true,
        Some(v) => !matches!(v.as_str(), "false" | "0" | "off"),
    }
}

fn get_attr(e: &quick_xml::events::BytesStart, name: &str) -> Option<String> {
    for attr in e.attributes().flatten() {
        let key = std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
        let local = key.rsplit(':').next().unwrap_or(key);
        if local == name {
            return Some(String::from_utf8_lossy(&attr.value).to_string());
        }
    }
    None
}
