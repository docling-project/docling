use serde::{Deserialize, Serialize};

use super::common::{ContentLayer, DocItemLabel};
use super::page::ProvenanceItem;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRef {
    pub mimetype: String,
    pub dpi: u32,
    pub size: ImageSize,
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSize {
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PictureMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefItem {
    #[serde(rename = "$ref")]
    pub ref_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PictureItem {
    pub self_ref: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<RefItem>,
    #[serde(default)]
    pub children: Vec<RefItem>,
    #[serde(default)]
    pub content_layer: ContentLayer,
    pub label: DocItemLabel,
    #[serde(default)]
    pub prov: Vec<ProvenanceItem>,
    #[serde(default)]
    pub captions: Vec<RefItem>,
    #[serde(default)]
    pub references: Vec<RefItem>,
    #[serde(default)]
    pub footnotes: Vec<RefItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<PictureMeta>,
    #[serde(default)]
    pub annotations: Vec<serde_json::Value>,
}
