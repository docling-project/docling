use std::path::Path;

use crate::models::document::DoclingDocument;

use super::Backend;

pub struct JsonDoclingBackend;

impl Backend for JsonDoclingBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let raw = std::fs::read_to_string(path)?;
        let content = raw.strip_prefix('\u{FEFF}').unwrap_or(&raw);
        let doc: DoclingDocument = serde_json::from_str(content)
            .map_err(|e| anyhow::anyhow!("Failed to parse Docling JSON '{}': {}", path.display(), e))?;
        Ok(doc)
    }
}
