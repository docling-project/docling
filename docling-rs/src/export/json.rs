use crate::models::document::DoclingDocument;

pub fn export(doc: &DoclingDocument) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(doc)?)
}
