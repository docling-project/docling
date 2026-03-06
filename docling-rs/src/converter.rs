use std::path::Path;

use crate::backend::asciidoc::AsciiDocBackend;
use crate::backend::csv::CsvBackend;
use crate::backend::docx::DocxBackend;
use crate::backend::html::HtmlBackend;
use crate::backend::image::ImageBackend;
use crate::backend::jats::JatsBackend;
use crate::backend::json_docling::JsonDoclingBackend;
use crate::backend::latex::LatexBackend;
use crate::backend::markdown::MarkdownBackend;
use crate::backend::mets_gbs::MetsGbsBackend;
use crate::backend::pdf::PdfBackend;
use crate::backend::pptx::PptxBackend;
use crate::backend::uspto::UsptoBackend;
use crate::backend::webvtt::WebVttBackend;
use crate::backend::xbrl::XbrlBackend;
use crate::backend::xlsx::XlsxBackend;
use crate::backend::Backend;
use crate::models::common::InputFormat;
use crate::models::document::DoclingDocument;

pub struct DocumentConverter;

impl Default for DocumentConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentConverter {
    pub fn new() -> Self {
        Self
    }

    pub fn convert(
        &self,
        path: &Path,
        input_format: Option<&InputFormat>,
    ) -> anyhow::Result<DoclingDocument> {
        let format = match input_format {
            Some(f) => f.clone(),
            None => InputFormat::from_extension(path)
                .ok_or_else(|| anyhow::anyhow!("Cannot detect format for: {}", path.display()))?,
        };

        let backend = self.get_backend(&format)?;
        backend.convert(path)
    }

    fn get_backend(&self, format: &InputFormat) -> anyhow::Result<Box<dyn Backend>> {
        match format {
            InputFormat::Csv => Ok(Box::new(CsvBackend)),
            InputFormat::Md => Ok(Box::new(MarkdownBackend)),
            InputFormat::Html => Ok(Box::new(HtmlBackend)),
            InputFormat::AsciiDoc => Ok(Box::new(AsciiDocBackend)),
            InputFormat::Vtt => Ok(Box::new(WebVttBackend)),
            InputFormat::Docx => Ok(Box::new(DocxBackend)),
            InputFormat::Pptx => Ok(Box::new(PptxBackend)),
            InputFormat::Xlsx => Ok(Box::new(XlsxBackend)),
            InputFormat::Pdf => Ok(Box::new(PdfBackend)),
            InputFormat::Latex => Ok(Box::new(LatexBackend)),
            InputFormat::XmlJats => Ok(Box::new(JatsBackend)),
            InputFormat::XmlUspto => Ok(Box::new(UsptoBackend)),
            InputFormat::JsonDocling => Ok(Box::new(JsonDoclingBackend)),
            InputFormat::XmlXbrl => Ok(Box::new(XbrlBackend)),
            InputFormat::Image => Ok(Box::new(ImageBackend)),
            InputFormat::MetsGbs => Ok(Box::new(MetsGbsBackend)),
            _ => anyhow::bail!("Backend for format '{}' is not yet implemented", format),
        }
    }
}
