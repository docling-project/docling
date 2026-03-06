use std::path::Path;

use base64::Engine;

use crate::models::common::{ContentLayer, DocItemLabel, InputFormat};
use crate::models::document::{create_doc_from_data, DoclingDocument};
use crate::models::picture::{ImageRef, ImageSize, PictureItem, RefItem};

use super::Backend;

pub struct ImageBackend;

impl Backend for ImageBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let data = std::fs::read(path)?;
        let mut doc = create_doc_from_data(path, &InputFormat::Image, &data);

        let image_ref = match image::load_from_memory(&data) {
            Ok(img) => {
                let (w, h) = (img.width(), img.height());
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("png")
                    .to_lowercase();
                let mimetype = match ext.as_str() {
                    "jpg" | "jpeg" => "image/jpeg",
                    "png" => "image/png",
                    "gif" => "image/gif",
                    "bmp" => "image/bmp",
                    "tif" | "tiff" => "image/tiff",
                    "webp" => "image/webp",
                    _ => "image/png",
                };

                let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
                let uri = format!("data:{};base64,{}", mimetype, b64);

                Some(ImageRef {
                    mimetype: mimetype.to_string(),
                    dpi: 72,
                    size: ImageSize {
                        width: w as f64,
                        height: h as f64,
                    },
                    uri,
                })
            }
            Err(e) => {
                log::warn!("Could not decode image {}: {}", path.display(), e);
                None
            }
        };

        let idx = doc.pictures.len();
        let self_ref = format!("#/pictures/{}", idx);

        doc.body.children.push(RefItem {
            ref_path: self_ref.clone(),
        });

        doc.pictures.push(PictureItem {
            self_ref,
            parent: Some(RefItem {
                ref_path: "#/body".to_string(),
            }),
            children: vec![],
            content_layer: ContentLayer::Body,
            label: DocItemLabel::Picture,
            prov: vec![],
            captions: vec![],
            references: vec![],
            footnotes: vec![],
            image: image_ref,
            meta: None,
            annotations: vec![],
        });

        doc.add_text(
            DocItemLabel::Text,
            "[Image content - OCR not available in this build]",
            None,
        );

        Ok(doc)
    }
}
