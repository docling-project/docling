use crate::models::document::DoclingDocument;

pub fn export(doc: &DoclingDocument) -> anyhow::Result<String> {
    Ok(serde_yaml::to_string(doc)?)
}
