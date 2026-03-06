use crate::models::document::DoclingDocument;

pub fn export(doc: &DoclingDocument) -> anyhow::Result<String> {
    let mut output = String::from("WEBVTT\n\n");
    let mut cue_num = 1u32;

    collect_cues(doc, "#/body", &mut output, &mut cue_num);

    Ok(output)
}

fn collect_cues(doc: &DoclingDocument, ref_path: &str, output: &mut String, cue_num: &mut u32) {
    if ref_path == "#/body" {
        for child in &doc.body.children {
            collect_cues(doc, &child.ref_path, output, cue_num);
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/texts/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(text_item) = doc.texts.get(idx) {
                if !text_item.text.is_empty() {
                    write_cue(output, *cue_num, &text_item.text);
                    *cue_num += 1;
                }
                for child in &text_item.children {
                    collect_cues(doc, &child.ref_path, output, cue_num);
                }
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/tables/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(table) = doc.tables.get(idx) {
                if let Some(ref grid) = table.data.grid {
                    let text: String = grid
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|c| c.text.as_str())
                                .collect::<Vec<_>>()
                                .join(" | ")
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.is_empty() {
                        write_cue(output, *cue_num, &text);
                        *cue_num += 1;
                    }
                }
            }
        }
        return;
    }

    if let Some(idx_str) = ref_path.strip_prefix("#/groups/") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if let Some(group) = doc.groups.get(idx) {
                for child in &group.children {
                    collect_cues(doc, &child.ref_path, output, cue_num);
                }
            }
        }
    }

    // pictures are skipped in VTT output
}

fn write_cue(output: &mut String, cue_num: u32, text: &str) {
    let start = format_time(cue_num as f64 - 1.0);
    let end = format_time(cue_num as f64);
    output.push_str(&format!("{}\n", cue_num));
    output.push_str(&format!("{} --> {}\n", start, end));
    output.push_str(&escape_cue_text(text));
    output.push_str("\n\n");
}

fn escape_cue_text(s: &str) -> String {
    s.replace("-->", "- ->")
}

fn format_time(seconds: f64) -> String {
    let total_secs = seconds.max(0.0) as u64;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    let millis = ((seconds.fract().abs()) * 1000.0) as u32;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
}
