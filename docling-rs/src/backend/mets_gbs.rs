use std::path::Path;

use super::Backend;
use crate::models::document::DoclingDocument;

pub struct MetsGbsBackend;

impl Backend for MetsGbsBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        anyhow::bail!(
            "METS/GBS backend is not yet fully implemented for: {}",
            path.display()
        )
    }
}
