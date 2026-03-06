use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

use super::common::{ContentLayer, DocItemLabel, GroupLabel, InputFormat};
use super::page::PageItem;
use super::picture::{ImageRef, PictureItem, RefItem};
use super::table::{TableCell, TableData, TableItem};
use super::text::{TextFormatting, TextItem};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentOrigin {
    pub mimetype: String,
    pub binary_hash: u64,
    pub filename: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupItem {
    pub self_ref: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<RefItem>,
    #[serde(default)]
    pub children: Vec<RefItem>,
    #[serde(default)]
    pub content_layer: ContentLayer,
    pub name: String,
    pub label: GroupLabel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoclingDocument {
    pub schema_name: String,
    pub version: String,
    pub name: String,
    pub origin: DocumentOrigin,
    pub furniture: GroupItem,
    pub body: GroupItem,
    #[serde(default)]
    pub groups: Vec<GroupItem>,
    #[serde(default)]
    pub texts: Vec<TextItem>,
    #[serde(default)]
    pub pictures: Vec<PictureItem>,
    #[serde(default)]
    pub tables: Vec<TableItem>,
    #[serde(default)]
    pub key_value_items: Vec<serde_json::Value>,
    #[serde(default)]
    pub form_items: Vec<serde_json::Value>,
    #[serde(default)]
    pub pages: BTreeMap<String, PageItem>,
}

impl DoclingDocument {
    pub fn new(name: &str, filename: &str, mimetype: &str, hash: u64) -> Self {
        Self {
            schema_name: "DoclingDocument".to_string(),
            version: "1.9.0".to_string(),
            name: name.to_string(),
            origin: DocumentOrigin {
                mimetype: mimetype.to_string(),
                binary_hash: hash,
                filename: filename.to_string(),
                uri: None,
            },
            furniture: GroupItem {
                self_ref: "#/furniture".to_string(),
                parent: None,
                children: vec![],
                content_layer: ContentLayer::Furniture,
                name: "_root_".to_string(),
                label: GroupLabel::Unspecified,
            },
            body: GroupItem {
                self_ref: "#/body".to_string(),
                parent: None,
                children: vec![],
                content_layer: ContentLayer::Body,
                name: "_root_".to_string(),
                label: GroupLabel::Unspecified,
            },
            groups: vec![],
            texts: vec![],
            pictures: vec![],
            tables: vec![],
            key_value_items: vec![],
            form_items: vec![],
            pages: BTreeMap::new(),
        }
    }

    pub fn add_text(
        &mut self,
        label: DocItemLabel,
        text: &str,
        parent_ref: Option<&str>,
    ) -> usize {
        let idx = self.texts.len();
        let self_ref = format!("#/texts/{}", idx);

        let parent;
        if let Some(parent_path) = parent_ref {
            parent = Some(RefItem { ref_path: parent_path.to_string() });
            self.add_child_ref(parent_path, &self_ref);
        } else {
            parent = Some(RefItem { ref_path: "#/body".to_string() });
            self.body.children.push(RefItem {
                ref_path: self_ref.clone(),
            });
        }

        self.texts.push(TextItem {
            self_ref,
            parent,
            children: vec![],
            content_layer: ContentLayer::Body,
            label,
            prov: vec![],
            orig: text.to_string(),
            text: text.to_string(),
            formatting: None,
            hyperlink: None,
            level: None,
            enumerated: None,
            marker: None,
            code_language: None,
        });
        idx
    }

    pub fn add_section_header(
        &mut self,
        text: &str,
        level: u32,
        parent_ref: Option<&str>,
    ) -> usize {
        let idx = self.texts.len();
        let self_ref = format!("#/texts/{}", idx);

        let parent;
        if let Some(parent_path) = parent_ref {
            parent = Some(RefItem { ref_path: parent_path.to_string() });
            self.add_child_ref(parent_path, &self_ref);
        } else {
            parent = Some(RefItem { ref_path: "#/body".to_string() });
            self.body.children.push(RefItem {
                ref_path: self_ref.clone(),
            });
        }

        self.texts.push(TextItem {
            self_ref,
            parent,
            children: vec![],
            content_layer: ContentLayer::Body,
            label: DocItemLabel::SectionHeader,
            prov: vec![],
            orig: text.to_string(),
            text: text.to_string(),
            formatting: None,
            hyperlink: None,
            level: Some(level),
            enumerated: None,
            marker: None,
            code_language: None,
        });
        idx
    }

    pub fn add_title(&mut self, text: &str, parent_ref: Option<&str>) -> usize {
        let idx = self.texts.len();
        let self_ref = format!("#/texts/{}", idx);

        let parent;
        if let Some(parent_path) = parent_ref {
            parent = Some(RefItem { ref_path: parent_path.to_string() });
            self.add_child_ref(parent_path, &self_ref);
        } else {
            parent = Some(RefItem { ref_path: "#/body".to_string() });
            self.body.children.push(RefItem {
                ref_path: self_ref.clone(),
            });
        }

        self.texts.push(TextItem {
            self_ref,
            parent,
            children: vec![],
            content_layer: ContentLayer::Body,
            label: DocItemLabel::Title,
            prov: vec![],
            orig: text.to_string(),
            text: text.to_string(),
            formatting: None,
            hyperlink: None,
            level: None,
            enumerated: None,
            marker: None,
            code_language: None,
        });
        idx
    }

    pub fn add_list_item(
        &mut self,
        text: &str,
        enumerated: bool,
        marker: Option<&str>,
        parent_ref: &str,
    ) -> usize {
        let idx = self.texts.len();
        let self_ref = format!("#/texts/{}", idx);

        self.add_child_ref(parent_ref, &self_ref);

        self.texts.push(TextItem {
            self_ref,
            parent: Some(RefItem {
                ref_path: parent_ref.to_string(),
            }),
            children: vec![],
            content_layer: ContentLayer::Body,
            label: DocItemLabel::ListItem,
            prov: vec![],
            orig: text.to_string(),
            text: text.to_string(),
            formatting: None,
            hyperlink: None,
            level: None,
            enumerated: Some(enumerated),
            marker: marker.map(|m| m.to_string()),
            code_language: None,
        });
        idx
    }

    pub fn add_group(
        &mut self,
        name: &str,
        label: GroupLabel,
        parent_ref: Option<&str>,
    ) -> usize {
        let idx = self.groups.len();
        let self_ref = format!("#/groups/{}", idx);

        let parent;
        if let Some(parent_path) = parent_ref {
            parent = Some(RefItem { ref_path: parent_path.to_string() });
            self.add_child_ref(parent_path, &self_ref);
        } else {
            parent = Some(RefItem { ref_path: "#/body".to_string() });
            self.body.children.push(RefItem {
                ref_path: self_ref.clone(),
            });
        }

        self.groups.push(GroupItem {
            self_ref,
            parent,
            children: vec![],
            content_layer: ContentLayer::Body,
            name: name.to_string(),
            label,
        });
        idx
    }

    pub fn tables_len(&self) -> usize {
        self.tables.len()
    }

    pub fn add_table(
        &mut self,
        cells: Vec<TableCell>,
        num_rows: u32,
        num_cols: u32,
        parent_ref: Option<&str>,
    ) -> usize {
        let idx = self.tables.len();
        let self_ref = format!("#/tables/{}", idx);

        let parent;
        if let Some(parent_path) = parent_ref {
            parent = Some(RefItem { ref_path: parent_path.to_string() });
            self.add_child_ref(parent_path, &self_ref);
        } else {
            parent = Some(RefItem { ref_path: "#/body".to_string() });
            self.body.children.push(RefItem {
                ref_path: self_ref.clone(),
            });
        }

        let mut data = TableData {
            table_cells: cells,
            num_rows,
            num_cols,
            grid: None,
        };
        data.build_grid();

        self.tables.push(TableItem {
            self_ref,
            parent,
            children: vec![],
            content_layer: ContentLayer::Body,
            label: DocItemLabel::Table,
            prov: vec![],
            captions: vec![],
            references: vec![],
            footnotes: vec![],
            data,
            annotations: vec![],
        });
        idx
    }

    pub fn add_picture(
        &mut self,
        alt_text: Option<&str>,
        parent_ref: Option<&str>,
    ) -> usize {
        let idx = self.pictures.len();
        let self_ref = format!("#/pictures/{}", idx);

        let parent;
        if let Some(parent_path) = parent_ref {
            parent = Some(RefItem { ref_path: parent_path.to_string() });
            self.add_child_ref(parent_path, &self_ref);
        } else {
            parent = Some(RefItem { ref_path: "#/body".to_string() });
            self.body.children.push(RefItem {
                ref_path: self_ref.clone(),
            });
        }

        self.pictures.push(PictureItem {
            self_ref,
            parent,
            children: vec![],
            content_layer: ContentLayer::Body,
            label: DocItemLabel::Picture,
            prov: vec![],
            captions: vec![],
            references: vec![],
            footnotes: vec![],
            image: None,
            meta: alt_text.map(|desc| super::picture::PictureMeta {
                description: Some(desc.to_string()),
                predicted_class: None,
                confidence: None,
            }),
            annotations: vec![],
        });
        idx
    }

    pub fn add_text_ext(
        &mut self,
        label: DocItemLabel,
        text: &str,
        parent_ref: Option<&str>,
        formatting: Option<TextFormatting>,
        hyperlink: Option<String>,
    ) -> usize {
        let idx = self.texts.len();
        let self_ref = format!("#/texts/{}", idx);

        let parent;
        if let Some(parent_path) = parent_ref {
            parent = Some(RefItem { ref_path: parent_path.to_string() });
            self.add_child_ref(parent_path, &self_ref);
        } else {
            parent = Some(RefItem { ref_path: "#/body".to_string() });
            self.body.children.push(RefItem {
                ref_path: self_ref.clone(),
            });
        }

        self.texts.push(TextItem {
            self_ref,
            parent,
            children: vec![],
            content_layer: ContentLayer::Body,
            label,
            prov: vec![],
            orig: text.to_string(),
            text: text.to_string(),
            formatting,
            hyperlink,
            level: None,
            enumerated: None,
            marker: None,
            code_language: None,
        });
        idx
    }

    pub fn add_furniture_text_to_parent(
        &mut self,
        label: DocItemLabel,
        text: &str,
        parent_ref: &str,
    ) -> usize {
        let idx = self.texts.len();
        let self_ref = format!("#/texts/{}", idx);

        self.add_child_ref(parent_ref, &self_ref);

        self.texts.push(TextItem {
            self_ref,
            parent: Some(RefItem {
                ref_path: parent_ref.to_string(),
            }),
            children: vec![],
            content_layer: ContentLayer::Furniture,
            label,
            prov: vec![],
            orig: text.to_string(),
            text: text.to_string(),
            formatting: None,
            hyperlink: None,
            level: None,
            enumerated: None,
            marker: None,
            code_language: None,
        });
        idx
    }

    pub fn add_furniture_text(&mut self, label: DocItemLabel, text: &str) -> usize {
        let idx = self.texts.len();
        let self_ref = format!("#/texts/{}", idx);

        self.furniture.children.push(RefItem {
            ref_path: self_ref.clone(),
        });

        self.texts.push(TextItem {
            self_ref,
            parent: Some(RefItem {
                ref_path: "#/furniture".to_string(),
            }),
            children: vec![],
            content_layer: ContentLayer::Furniture,
            label,
            prov: vec![],
            orig: text.to_string(),
            text: text.to_string(),
            formatting: None,
            hyperlink: None,
            level: None,
            enumerated: None,
            marker: None,
            code_language: None,
        });
        idx
    }

    pub fn set_picture_image(&mut self, idx: usize, image_ref: ImageRef) {
        if idx < self.pictures.len() {
            self.pictures[idx].image = Some(image_ref);
        }
    }

    fn add_child_ref(&mut self, parent_path: &str, child_ref: &str) {
        let child = RefItem {
            ref_path: child_ref.to_string(),
        };

        if parent_path == "#/body" {
            self.body.children.push(child);
            return;
        }
        if parent_path == "#/furniture" {
            self.furniture.children.push(child);
            return;
        }

        if let Some(idx_str) = parent_path.strip_prefix("#/texts/") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if idx < self.texts.len() {
                    self.texts[idx].children.push(child);
                    return;
                }
            }
        }
        if let Some(idx_str) = parent_path.strip_prefix("#/groups/") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if idx < self.groups.len() {
                    self.groups[idx].children.push(child);
                    return;
                }
            }
        }
        if let Some(idx_str) = parent_path.strip_prefix("#/tables/") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if idx < self.tables.len() {
                    self.tables[idx].children.push(child);
                    return;
                }
            }
        }
        if let Some(idx_str) = parent_path.strip_prefix("#/pictures/") {
            if let Ok(idx) = idx_str.parse::<usize>() {
                if idx < self.pictures.len() {
                    self.pictures[idx].children.push(child);
                    return;
                }
            }
        }

        log::warn!(
            "Unknown parent ref '{}' when adding child '{}'; child not linked",
            parent_path,
            child_ref
        );
    }
}

pub fn compute_hash(data: &[u8]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let bytes: [u8; 8] = result[..8].try_into().unwrap();
    u64::from_be_bytes(bytes)
}

pub fn doc_name_from_path(path: &Path) -> String {
    let raw = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    sanitize_name(raw)
}

fn sanitize_name(name: &str) -> String {
    let sanitized: String = name
        .replace(['/', '\\'], "_")
        .replace("..", "_");
    if sanitized.is_empty() || sanitized == "." {
        "unknown".to_string()
    } else {
        sanitized
    }
}

pub fn create_doc_from_file(path: &Path, format: &InputFormat) -> anyhow::Result<DoclingDocument> {
    let data = std::fs::read(path)?;
    Ok(create_doc_from_data(path, format, &data))
}

pub fn create_doc_from_data(path: &Path, format: &InputFormat, data: &[u8]) -> DoclingDocument {
    let hash = compute_hash(data);
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let name = doc_name_from_path(path);
    DoclingDocument::new(&name, filename, format.mimetype(), hash)
}
