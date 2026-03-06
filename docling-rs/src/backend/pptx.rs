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
use super::Backend;

pub struct PptxBackend;

impl Backend for PptxBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::Pptx)?;
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open PPTX file: {}", path.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("Invalid ZIP/PPTX file: {}", path.display()))?;

        let slide_size = read_slide_size(&mut archive);
        let media = preload_media(&mut archive);
        let slide_names = find_slides(&mut archive);

        for (slide_idx, slide_name) in slide_names.iter().enumerate() {
            let rels_path = slide_name
                .replace("ppt/slides/", "ppt/slides/_rels/")
                + ".rels";
            let rels = parse_relationships(&mut archive, &rels_path);
            let xml = read_zip_entry(&mut archive, slide_name)?;

            let slide_group_idx = doc.add_group(
                &format!("slide-{}", slide_idx),
                GroupLabel::Chapter,
                None,
            );
            let slide_parent = format!("#/groups/{}", slide_group_idx);

            if let Some((w, h)) = slide_size {
                let page_no = (slide_idx + 1) as u32;
                doc.pages.insert(
                    page_no.to_string(),
                    crate::models::page::PageItem {
                        size: crate::models::page::Size {
                            width: w,
                            height: h,
                        },
                        page_no,
                        image: None,
                    },
                );
            }

            parse_slide_xml(
                &xml,
                slide_idx,
                &rels,
                &media,
                &slide_parent,
                &mut doc,
            )?;

            // Parse notes for this slide via relationship lookup
            if let Some(notes_target) = rels.resolve_notes_path() {
                if let Ok(notes_xml) = read_zip_entry(&mut archive, &notes_target) {
                    parse_notes_xml(&notes_xml, &slide_parent, &mut doc);
                }
            }
        }

        Ok(doc)
    }
}

// ---------------------------------------------------------------------------
// Presentation-level info
// ---------------------------------------------------------------------------

fn read_slide_size(
    archive: &mut zip::ZipArchive<std::fs::File>,
) -> Option<(f64, f64)> {
    let xml = read_zip_entry(archive, "ppt/presentation.xml").ok()?;
    let mut reader = Reader::from_str(&xml);
    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local = get_local_name(e);
                if local == "sldSz" {
                    let cx = get_attr(e, "cx")
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(9144000.0);
                    let cy = get_attr(e, "cy")
                        .and_then(|v| v.parse::<f64>().ok())
                        .unwrap_or(6858000.0);
                    // EMU to points (1 pt = 12700 EMU)
                    return Some((cx / 12700.0, cy / 12700.0));
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Relationships and media
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Relationships {
    map: HashMap<String, (String, bool)>,
}

impl Relationships {
    fn resolve_url(&self, r_id: &str) -> Option<String> {
        self.map.get(r_id).and_then(|(target, is_ext)| {
            if *is_ext {
                Some(target.clone())
            } else {
                None
            }
        })
    }

    fn resolve_notes_path(&self) -> Option<String> {
        self.map.values().find_map(|(target, is_ext)| {
            if !is_ext && target.contains("notesSlides/") {
                Some(format!("ppt/slides/{}", target).replace("slides/../", ""))
            } else {
                None
            }
        })
    }

    fn resolve_media_path(&self, r_id: &str) -> Option<String> {
        self.map.get(r_id).and_then(|(target, is_ext)| {
            if *is_ext {
                None
            } else if target.starts_with('/') {
                Some(target.trim_start_matches('/').to_string())
            } else {
                Some(format!("ppt/slides/{}", target).replace("slides/../", ""))
            }
        })
    }
}

fn parse_relationships(
    archive: &mut zip::ZipArchive<std::fs::File>,
    path: &str,
) -> Relationships {
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
                    if let (Some(id), Some(target)) =
                        (get_attr(e, "Id"), get_attr(e, "Target"))
                    {
                        let is_ext = get_attr(e, "TargetMode")
                            .map_or(false, |m| m.eq_ignore_ascii_case("External"));
                        rels.map.insert(id, (target, is_ext));
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

fn preload_media(archive: &mut zip::ZipArchive<std::fs::File>) -> HashMap<String, Vec<u8>> {
    let mut media = HashMap::new();
    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
        .filter(|n| n.starts_with("ppt/media/"))
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

fn find_slides(archive: &mut zip::ZipArchive<std::fs::File>) -> Vec<String> {
    let mut slides = Vec::new();
    for i in 0..archive.len() {
        if let Ok(file) = archive.by_index(i) {
            let name = file.name().to_string();
            if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                slides.push(name);
            }
        }
    }
    slides.sort_by_key(|s| {
        s.trim_start_matches("ppt/slides/slide")
            .trim_end_matches(".xml")
            .parse::<u32>()
            .unwrap_or(0)
    });
    slides
}

// ---------------------------------------------------------------------------
// Slide parser
// ---------------------------------------------------------------------------

const SKIP_PLACEHOLDER_TYPES: &[&str] = &["ftr", "dt", "sldNum"];

#[derive(Clone)]
struct ParaInfo {
    text: String,
    is_bullet: bool,
    is_auto_num: bool,
    is_bu_none: bool,
    is_bu_blip: bool,
}

#[derive(Default)]
struct RunFmt {
    bold: bool,
    italic: bool,
    underline: bool,
    strikethrough: bool,
}

impl RunFmt {}

#[allow(clippy::too_many_arguments)]
fn parse_slide_xml(
    xml: &str,
    _slide_idx: usize,
    rels: &Relationships,
    media: &HashMap<String, Vec<u8>>,
    slide_parent: &str,
    doc: &mut DoclingDocument,
) -> anyhow::Result<()> {
    let mut reader = Reader::from_str(xml);
    let mut texts: Vec<ParaInfo> = Vec::new();
    let mut current_para = String::new();
    let mut in_paragraph = false;
    let mut in_run = false;

    let mut in_shape = false;
    let mut shape_is_title = false;
    let mut shape_is_subtitle = false;
    let mut shape_is_body = false;
    let mut shape_skip = false;

    // Table state
    let mut table_cells: Vec<TableCell> = Vec::new();
    let mut in_table = false;
    let mut table_row: u32 = 0;
    let mut table_col: u32 = 0;
    let mut table_max_cols: u32 = 0;
    let mut in_table_cell = false;
    let mut cell_text = String::new();
    let mut cell_col_span: u32 = 1;
    let mut cell_row_span: u32 = 1;
    let mut cell_is_hmerge = false;
    let mut cell_is_vmerge = false;

    // Run formatting
    let mut in_rpr = false;
    let mut run_fmt = RunFmt::default();
    let mut para_fmt = RunFmt::default();
    let mut para_run_count: u32 = 0;
    let mut _para_has_mixed_fmt = false;

    // Hyperlink
    let mut run_hyperlink: Option<String> = None;
    let mut _para_hyperlink: Option<String> = None;

    // List detection (per paragraph)
    let mut para_bullet = false;
    let mut para_auto_num = false;
    let mut para_bu_none = false;
    let mut para_bu_blip = false;
    let mut _para_indent_level: u32 = 0;

    // Picture (p:pic) tracking
    let mut in_pic = false;
    let mut pic_blip_rid: Option<String> = None;
    let mut pic_desc: Option<String> = None;

    // Group shape (p:grpSp) depth tracking
    let mut grp_depth: u32 = 0;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let local = get_local_name(e);
                match local.as_str() {
                    "sp" => {
                        if !texts.is_empty() && in_shape && !shape_skip {
                            emit_txbody_texts(
                                &texts,
                                shape_is_title,
                                shape_is_subtitle,
                                shape_is_body,
                                slide_parent,
                                doc,
                            );
                            texts.clear();
                        }
                        in_shape = true;
                        shape_is_title = false;
                        shape_is_subtitle = false;
                        shape_is_body = false;
                        shape_skip = false;
                    }
                    "grpSp" => grp_depth += 1,
                    "pic" if !in_table => {
                        in_pic = true;
                        pic_blip_rid = None;
                        pic_desc = None;
                    }
                    "tbl" => {
                        in_table = true;
                        table_cells.clear();
                        table_row = 0;
                        table_max_cols = 0;
                    }
                    "tr" if in_table => table_col = 0,
                    "tc" if in_table => {
                        in_table_cell = true;
                        cell_text.clear();
                        cell_col_span = get_attr(e, "gridSpan")
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(1);
                        cell_row_span = get_attr(e, "rowSpan")
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(1);
                        cell_is_hmerge = false;
                        cell_is_vmerge = false;
                        check_merge_attrs(e, &mut cell_is_hmerge, &mut cell_is_vmerge);
                    }
                    "tcPr" if in_table_cell => {
                        check_merge_attrs(e, &mut cell_is_hmerge, &mut cell_is_vmerge);
                    }
                    "p" => {
                        in_paragraph = true;
                        current_para.clear();
                        para_bullet = false;
                        para_auto_num = false;
                        para_bu_none = false;
                        para_bu_blip = false;
                        _para_indent_level = 0;
                        para_fmt = RunFmt::default();
                        para_run_count = 0;
                        _para_has_mixed_fmt = false;
                        _para_hyperlink = None;
                    }
                    "pPr" if in_paragraph => {
                        if let Some(lvl) = get_attr(e, "lvl") {
                            _para_indent_level = lvl.parse().unwrap_or(0);
                        }
                    }
                    "r" if in_paragraph => {
                        in_run = true;
                        run_fmt = RunFmt::default();
                        run_hyperlink = None;
                    }
                    "rPr" if in_run => {
                        in_rpr = true;
                        if let Some(b) = get_attr(e, "b") {
                            run_fmt.bold = b == "1" || b == "true";
                        }
                        if let Some(i) = get_attr(e, "i") {
                            run_fmt.italic = i == "1" || i == "true";
                        }
                        if let Some(u) = get_attr(e, "u") {
                            run_fmt.underline = u != "none";
                        }
                        if let Some(s) = get_attr(e, "strike") {
                            run_fmt.strikethrough = s != "noStrike";
                        }
                    }
                    "br" if in_paragraph => {
                        if in_table_cell {
                            cell_text.push(' ');
                        } else {
                            current_para.push(' ');
                        }
                    }
                    "hlinkClick" if in_rpr || in_run => {
                        if let Some(rid) = get_attr(e, "id") {
                            run_hyperlink = rels.resolve_url(&rid);
                        }
                    }
                    "blip" if in_pic => {
                        pic_blip_rid = get_attr(e, "embed")
                            .or_else(|| get_attr(e, "link"));
                    }
                    "cNvPr" if in_pic => {
                        if pic_desc.is_none() {
                            pic_desc = get_attr(e, "descr");
                        }
                    }
                    "tab" if in_paragraph && !shape_skip => {
                        if in_table_cell {
                            cell_text.push(' ');
                        } else {
                            current_para.push(' ');
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let local = get_local_name(e);
                match local.as_str() {
                    "ph" if in_shape => {
                        let ph_type = get_attr(e, "type").unwrap_or_default();
                        if ph_type == "title" || ph_type == "ctrTitle" {
                            shape_is_title = true;
                        } else if ph_type == "subTitle" {
                            shape_is_subtitle = true;
                        } else if ph_type == "body" || ph_type == "obj" || ph_type.is_empty() {
                            shape_is_body = true;
                        }
                        if SKIP_PLACEHOLDER_TYPES.contains(&ph_type.as_str()) {
                            shape_skip = true;
                        }
                    }
                    "buChar" if in_paragraph => para_bullet = true,
                    "buAutoNum" if in_paragraph => para_auto_num = true,
                    "buBlip" if in_paragraph => para_bu_blip = true,
                    "buNone" if in_paragraph => {
                        para_bullet = false;
                        para_auto_num = false;
                        para_bu_blip = false;
                        para_bu_none = true;
                    }
                    "br" if in_paragraph => {
                        if in_table_cell {
                            cell_text.push(' ');
                        } else {
                            current_para.push(' ');
                        }
                    }
                    "tab" if in_paragraph && !shape_skip => {
                        if in_table_cell {
                            cell_text.push(' ');
                        } else {
                            current_para.push(' ');
                        }
                    }
                    "tcPr" if in_table_cell => {
                        check_merge_attrs(e, &mut cell_is_hmerge, &mut cell_is_vmerge);
                    }
                    "blip" if in_pic => {
                        pic_blip_rid = get_attr(e, "embed")
                            .or_else(|| get_attr(e, "link"));
                    }
                    "cNvPr" if in_pic => {
                        pic_desc = get_attr(e, "descr");
                    }
                    "hlinkClick" if in_run => {
                        if let Some(rid) = get_attr(e, "id") {
                            run_hyperlink = rels.resolve_url(&rid);
                        }
                    }
                    "rPr" if in_run => {
                        if let Some(b) = get_attr(e, "b") {
                            run_fmt.bold = b == "1" || b == "true";
                        }
                        if let Some(i) = get_attr(e, "i") {
                            run_fmt.italic = i == "1" || i == "true";
                        }
                        if let Some(u) = get_attr(e, "u") {
                            run_fmt.underline = u != "none";
                        }
                        if let Some(s) = get_attr(e, "strike") {
                            run_fmt.strikethrough = s != "noStrike";
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_run && in_paragraph && !shape_skip {
                    let t = String::from_utf8_lossy(e.as_ref()).to_string();
                    if in_table_cell {
                        cell_text.push_str(&t);
                    } else {
                        current_para.push_str(&t);
                    }
                    if run_hyperlink.is_some() {
                        _para_hyperlink = run_hyperlink.clone();
                    }
                }
            }
            Ok(Event::GeneralRef(ref e)) => {
                if in_run && in_paragraph && !shape_skip {
                    let entity = String::from_utf8_lossy(e.as_ref());
                    let ch = match entity.as_ref() {
                        "amp" => "&", "lt" => "<", "gt" => ">",
                        "apos" => "'", "quot" => "\"", _ => "",
                    };
                    if !ch.is_empty() {
                        if in_table_cell {
                            cell_text.push_str(ch);
                        } else {
                            current_para.push_str(ch);
                        }
                    }
                }
            }
            Ok(Event::End(ref e)) => {
                let local = get_local_name_end(e);
                match local.as_str() {
                    "rPr" => in_rpr = false,
                    "r" => {
                        if in_run {
                            para_run_count += 1;
                            if para_run_count == 1 {
                                para_fmt = RunFmt {
                                    bold: run_fmt.bold,
                                    italic: run_fmt.italic,
                                    underline: run_fmt.underline,
                                    strikethrough: run_fmt.strikethrough,
                                };
                            } else if para_fmt.bold != run_fmt.bold
                                || para_fmt.italic != run_fmt.italic
                                || para_fmt.underline != run_fmt.underline
                                || para_fmt.strikethrough != run_fmt.strikethrough
                            {
                                _para_has_mixed_fmt = true;
                            }
                        }
                        in_run = false;
                    }
                    "p" => {
                        in_paragraph = false;
                        if in_table_cell {
                            if !cell_text.is_empty() && !current_para.is_empty() {
                                cell_text.push(' ');
                            }
                            cell_text.push_str(&current_para);
                        } else if !shape_skip {
                            texts.push(ParaInfo {
                                text: current_para.clone(),
                                is_bullet: para_bullet || para_bu_blip,
                                is_auto_num: para_auto_num,
                                is_bu_none: para_bu_none,
                                is_bu_blip: para_bu_blip,
                            });
                        }
                    }
                    "tc" if in_table => {
                        in_table_cell = false;
                        let trimmed = cell_text.trim().to_string();
                        if !cell_is_hmerge && !cell_is_vmerge && !trimmed.is_empty() {
                            table_cells.push(TableCell {
                                row_span: cell_row_span,
                                col_span: cell_col_span,
                                start_row_offset_idx: table_row,
                                end_row_offset_idx: table_row + cell_row_span,
                                start_col_offset_idx: table_col,
                                end_col_offset_idx: table_col + cell_col_span,
                                text: trimmed,
                                column_header: table_row == 0,
                                row_header: false,
                                row_section: false,
                                fillable: false,
                                formatted_text: None,
                            });
                        }
                        table_col += 1;
                    }
                    "tr" if in_table => {
                        table_max_cols = table_max_cols.max(table_col);
                        table_row += 1;
                    }
                    "tbl" => {
                        in_table = false;
                        if !table_cells.is_empty() {
                            doc.add_table(
                                table_cells.clone(),
                                table_row,
                                table_max_cols,
                                Some(slide_parent),
                            );
                        }
                    }
                    "pic" if in_pic => {
                        in_pic = false;
                        if let Some(ref rid) = pic_blip_rid {
                            emit_image(
                                rid,
                                pic_desc.as_deref(),
                                rels,
                                media,
                                doc,
                                Some(slide_parent),
                            );
                        }
                    }
                    "txBody" => {
                        if !texts.is_empty() && !shape_skip {
                            emit_txbody_texts(
                                &texts,
                                shape_is_title,
                                shape_is_subtitle,
                                shape_is_body,
                                slide_parent,
                                doc,
                            );
                            texts.clear();
                        }
                    }
                    "sp" => {
                        if !texts.is_empty() && !shape_skip {
                            for pi in &texts {
                                if !pi.text.trim().is_empty() {
                                    doc.add_text_ext(
                                        DocItemLabel::Text,
                                        pi.text.trim(),
                                        Some(slide_parent),
                                        None,
                                        None,
                                    );
                                }
                            }
                            texts.clear();
                        }
                        in_shape = false;
                        shape_is_title = false;
                        shape_is_subtitle = false;
                        shape_is_body = false;
                        shape_skip = false;
                    }
                    "grpSp" => {
                        grp_depth = grp_depth.saturating_sub(1);
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(anyhow::anyhow!("XML parse error in slide: {}", e));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Emit paragraphs collected from a single `<p:txBody>`, handling titles,
/// per-paragraph list detection, and body-placeholder bullet fallback.
fn emit_txbody_texts(
    texts: &[ParaInfo],
    shape_is_title: bool,
    shape_is_subtitle: bool,
    shape_is_body: bool,
    slide_parent: &str,
    doc: &mut DoclingDocument,
) {
    if shape_is_title {
        for pi in texts {
            if !pi.text.trim().is_empty() {
                doc.add_title(pi.text.trim(), Some(slide_parent));
            }
        }
        return;
    }

    if shape_is_subtitle {
        for pi in texts {
            if !pi.text.trim().is_empty() {
                doc.add_section_header(pi.text.trim(), 1, Some(slide_parent));
            }
        }
        return;
    }

    // Per-paragraph list detection:
    // For body/obj placeholders, paragraphs are list items unless buNone is set.
    // For non-placeholder shapes, paragraphs are list items only with explicit buChar/buAutoNum.
    let mut list_group_ref: Option<String> = None;
    let mut list_is_numbered = false;
    let mut enum_counter: u32 = 0;

    for pi in texts {
        let is_list = if pi.is_bullet || pi.is_auto_num || pi.is_bu_blip {
            true
        } else if pi.is_bu_none {
            false
        } else {
            shape_is_body
        };
        let is_numbered = pi.is_auto_num;

        if is_list {
            if list_group_ref.is_none() || list_is_numbered != is_numbered {
                let label = if is_numbered {
                    GroupLabel::OrderedList
                } else {
                    GroupLabel::List
                };
                let gidx = doc.add_group("list", label, Some(slide_parent));
                list_group_ref = Some(format!("#/groups/{}", gidx));
                list_is_numbered = is_numbered;
                enum_counter = 0;
            }

            let marker = if is_numbered {
                enum_counter += 1;
                format!("{}.", enum_counter)
            } else {
                "-".to_string()
            };

            doc.add_list_item(
                &pi.text,
                is_numbered,
                Some(&marker),
                list_group_ref.as_deref().unwrap(),
            );
        } else {
            list_group_ref = None;
            enum_counter = 0;

            if !pi.text.trim().is_empty() {
                doc.add_text_ext(
                    DocItemLabel::Paragraph,
                    pi.text.trim(),
                    Some(slide_parent),
                    None,
                    None,
                );
            }
        }
    }
}

fn check_merge_attrs(
    e: &quick_xml::events::BytesStart,
    hmerge: &mut bool,
    vmerge: &mut bool,
) {
    if let Some(hm) = get_attr(e, "hMerge") {
        if hm == "1" || hm == "true" {
            *hmerge = true;
        }
    }
    if let Some(vm) = get_attr(e, "vMerge") {
        if vm == "1" || vm == "true" {
            *vmerge = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Notes parser
// ---------------------------------------------------------------------------

fn parse_notes_xml(xml: &str, slide_parent: &str, doc: &mut DoclingDocument) {
    let mut reader = Reader::from_str(xml);
    let mut in_paragraph = false;
    let mut para_text = String::new();
    let mut in_run = false;
    let mut all_text = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local = get_local_name(e);
                match local.as_str() {
                    "p" => {
                        in_paragraph = true;
                        para_text.clear();
                    }
                    "r" if in_paragraph => in_run = true,
                    "br" if in_paragraph => {
                        para_text.push('\n');
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) if in_run && in_paragraph => {
                para_text.push_str(&String::from_utf8_lossy(e.as_ref()));
            }
            Ok(Event::GeneralRef(ref e)) if in_run && in_paragraph => {
                let entity = String::from_utf8_lossy(e.as_ref());
                let ch = match entity.as_ref() {
                    "amp" => "&", "lt" => "<", "gt" => ">",
                    "apos" => "'", "quot" => "\"", _ => "",
                };
                para_text.push_str(ch);
            }
            Ok(Event::End(ref e)) => {
                let local = get_local_name_end(e);
                match local.as_str() {
                    "r" => in_run = false,
                    "p" => {
                        in_paragraph = false;
                        let text = para_text.trim();
                        if !text.is_empty() {
                            if !all_text.is_empty() {
                                all_text.push('\n');
                            }
                            all_text.push_str(text);
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

    let trimmed = all_text.trim();
    if !trimmed.is_empty() {
        doc.add_furniture_text_to_parent(DocItemLabel::Text, trimmed, slide_parent);
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
        None => return,
    };
    let data = match media.get(&media_path) {
        Some(d) => d,
        None => return,
    };

    let idx = doc.add_picture(alt_text, parent_ref);

    let ext = media_path
        .rsplit('.')
        .next()
        .unwrap_or("png")
        .to_lowercase();
    let mimetype = match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "emf" => "image/x-emf",
        "wmf" => "image/x-wmf",
        "svg" => "image/svg+xml",
        _ => "image/png",
    };

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

// ---------------------------------------------------------------------------
// Low-level XML helpers
// ---------------------------------------------------------------------------

fn read_zip_entry(
    archive: &mut zip::ZipArchive<std::fs::File>,
    name: &str,
) -> anyhow::Result<String> {
    let mut file = archive
        .by_name(name)
        .with_context(|| format!("Missing entry '{}' in PPTX", name))?;
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
