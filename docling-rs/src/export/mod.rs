pub mod csv;
pub mod doctags;
pub mod html;
pub mod json;
pub mod markdown;
pub mod text;
pub mod vtt;
pub mod yaml;

use crate::models::common::{ImageRefMode, OutputFormat};
use crate::models::document::DoclingDocument;

pub fn export_document(
    doc: &DoclingDocument,
    format: &OutputFormat,
    image_mode: Option<ImageRefMode>,
) -> anyhow::Result<String> {
    let mode = image_mode.unwrap_or(ImageRefMode::Placeholder);
    match format {
        OutputFormat::Json => json::export(doc),
        OutputFormat::Yaml => yaml::export(doc),
        OutputFormat::Markdown => markdown::export(doc, mode),
        OutputFormat::Text => text::export(doc),
        OutputFormat::Csv => csv::export(doc),
        OutputFormat::Html => html::export(doc, mode),
        OutputFormat::DocTags => doctags::export(doc),
        OutputFormat::Vtt => vtt::export(doc),
    }
}
