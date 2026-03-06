use crate::models::common::{DocItemLabel, GroupLabel};
use crate::models::document::DoclingDocument;

pub fn export(doc: &DoclingDocument) -> anyhow::Result<String> {
    let mut output = String::new();
    output.push_str("<doctags>\n");
    export_node(doc, "#/body", &mut output);
    if !doc.furniture.children.is_empty() {
        for child in &doc.furniture.children {
            export_node(doc, &child.ref_path, &mut output);
        }
    }
    output.push_str("</doctags>\n");
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
                let tag = label_to_tag(&text_item.label, text_item.level);
                output.push_str(&format!(
                    "<{}>{}</{}>\n",
                    tag,
                    xml_escape(&text_item.text),
                    tag
                ));
                for child in &text_item.children {
                    export_node(doc, &child.ref_path, output);
                }
            } else {
                log::debug!("DocTags: text index {} out of bounds", idx);
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/tables/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(table) = doc.tables.get(idx) {
                output.push_str("<table>\n");
                if let Some(ref grid) = table.data.grid {
                    for row in grid {
                        output.push_str("<tr>");
                        for cell in row {
                            let tag = if cell.column_header { "th" } else { "td" };
                            output.push_str(&format!(
                                "<{}>{}</{}>",
                                tag,
                                xml_escape(&cell.text),
                                tag
                            ));
                        }
                        output.push_str("</tr>\n");
                    }
                }
                output.push_str("</table>\n");
            } else {
                log::debug!("DocTags: table index {} out of bounds", idx);
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/groups/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(group) = doc.groups.get(idx) {
                let tag = group_label_to_tag(&group.label);
                output.push_str(&format!("<{}>\n", tag));
                for child in &group.children {
                    export_node(doc, &child.ref_path, output);
                }
                output.push_str(&format!("</{}>\n", tag));
            } else {
                log::debug!("DocTags: group index {} out of bounds", idx);
            }
        }
        return;
    }

    if ref_path.starts_with("#/pictures/") {
        output.push_str("<picture/>\n");
        return;
    }

    log::debug!("DocTags: unknown ref_path '{}'", ref_path);
}

fn label_to_tag(label: &DocItemLabel, level: Option<u32>) -> String {
    match label {
        DocItemLabel::Title => "title".to_string(),
        DocItemLabel::SectionHeader => {
            let lvl = level.unwrap_or(1);
            format!("section_header_level_{}", lvl)
        }
        DocItemLabel::Paragraph | DocItemLabel::Text => "text".to_string(),
        DocItemLabel::ListItem => "list_item".to_string(),
        DocItemLabel::Code => "code".to_string(),
        DocItemLabel::Formula => "formula".to_string(),
        DocItemLabel::Caption => "caption".to_string(),
        DocItemLabel::Reference => "reference".to_string(),
        DocItemLabel::Footnote => "footnote".to_string(),
        DocItemLabel::PageHeader => "page_header".to_string(),
        DocItemLabel::PageFooter => "page_footer".to_string(),
        _ => "text".to_string(),
    }
}

fn group_label_to_tag(label: &GroupLabel) -> &'static str {
    match label {
        GroupLabel::List => "list",
        GroupLabel::OrderedList => "ordered_list",
        GroupLabel::Chapter => "chapter",
        GroupLabel::Section => "section",
        GroupLabel::Sheet => "sheet",
        GroupLabel::Slide => "slide",
        GroupLabel::FormArea => "form_area",
        GroupLabel::KeyValueArea => "key_value_area",
        GroupLabel::CommentSection => "comment_section",
        GroupLabel::Inline => "inline",
        GroupLabel::PictureArea => "picture_area",
        GroupLabel::Unspecified | GroupLabel::Unknown => "group",
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
