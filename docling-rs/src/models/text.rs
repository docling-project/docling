use serde::{Deserialize, Serialize};

use super::common::{ContentLayer, DocItemLabel};
use super::page::ProvenanceItem;
use super::picture::RefItem;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextFormatting {
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bold: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub italic: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub underline: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub strikethrough: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextItem {
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
    pub orig: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formatting: Option<TextFormatting>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperlink: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enumerated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub marker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_language: Option<String>,
}
