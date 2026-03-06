use serde::{Deserialize, Serialize};

use super::picture::ImageRef;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub l: f64,
    pub t: f64,
    pub r: f64,
    pub b: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coord_origin: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceItem {
    pub page_no: u32,
    pub bbox: BoundingBox,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub charspan: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageItem {
    pub size: Size,
    pub page_no: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageRef>,
}
