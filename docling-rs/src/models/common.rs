use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputFormat {
    Csv,
    Md,
    Html,
    #[serde(rename = "asciidoc")]
    AsciiDoc,
    #[serde(rename = "vtt")]
    Vtt,
    Docx,
    Pptx,
    Xlsx,
    Pdf,
    Image,
    Latex,
    #[serde(rename = "xml_jats")]
    XmlJats,
    #[serde(rename = "xml_uspto")]
    XmlUspto,
    #[serde(rename = "xml_xbrl")]
    XmlXbrl,
    #[serde(rename = "json_docling")]
    JsonDocling,
    #[serde(rename = "mets_gbs")]
    MetsGbs,
    Audio,
}

impl fmt::Display for InputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Csv => write!(f, "csv"),
            Self::Md => write!(f, "md"),
            Self::Html => write!(f, "html"),
            Self::AsciiDoc => write!(f, "asciidoc"),
            Self::Vtt => write!(f, "vtt"),
            Self::Docx => write!(f, "docx"),
            Self::Pptx => write!(f, "pptx"),
            Self::Xlsx => write!(f, "xlsx"),
            Self::Pdf => write!(f, "pdf"),
            Self::Image => write!(f, "image"),
            Self::Latex => write!(f, "latex"),
            Self::XmlJats => write!(f, "xml_jats"),
            Self::XmlUspto => write!(f, "xml_uspto"),
            Self::XmlXbrl => write!(f, "xml_xbrl"),
            Self::JsonDocling => write!(f, "json_docling"),
            Self::MetsGbs => write!(f, "mets_gbs"),
            Self::Audio => write!(f, "audio"),
        }
    }
}

impl InputFormat {
    pub fn from_extension(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "csv" => Some(Self::Csv),
            "md" | "markdown" => Some(Self::Md),
            "html" | "htm" | "xhtml" => Some(Self::Html),
            "asciidoc" | "adoc" | "asc" => Some(Self::AsciiDoc),
            "vtt" => Some(Self::Vtt),
            "docx" => Some(Self::Docx),
            "pptx" | "potx" | "ppsx" | "pptm" | "potm" | "ppsm" => Some(Self::Pptx),
            "xlsx" | "xlsm" | "xls" => Some(Self::Xlsx),
            "pdf" => Some(Self::Pdf),
            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "tif" | "tiff" | "webp" => {
                Some(Self::Image)
            }
            "tex" | "latex" => Some(Self::Latex),
            "nxml" => Some(Self::XmlJats),
            "xml" => Self::sniff_xml_format(path),
            "json" => Some(Self::JsonDocling),
            "mp3" | "wav" | "flac" | "ogg" | "m4a" => Some(Self::Audio),
            _ => None,
        }
    }

    fn sniff_xml_format(path: &Path) -> Option<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).ok()?;
        let mut buf = vec![0u8; 4096];
        let n = file.read(&mut buf).ok()?;
        buf.truncate(n);
        let prefix = String::from_utf8_lossy(&buf);
        let lower = prefix.to_lowercase();

        if lower.contains("<us-patent-application")
            || lower.contains("<us-patent-grant")
            || lower.contains("<patent-application-publication")
            || lower.contains("<patdoc")
            || lower.contains("<!doctype us-patent")
        {
            Some(Self::XmlUspto)
        } else if lower.contains("<article")
            || lower.contains("jats")
            || lower.contains("nlm-articleset")
        {
            Some(Self::XmlJats)
        } else if lower.contains("xbrl") || lower.contains("<xbrli:") {
            Some(Self::XmlXbrl)
        } else {
            None
        }
    }

    pub fn mimetype(&self) -> &'static str {
        match self {
            Self::Csv => "text/csv",
            Self::Md => "text/markdown",
            Self::Html => "text/html",
            Self::AsciiDoc => "text/asciidoc",
            Self::Vtt => "text/vtt",
            Self::Docx => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            Self::Pptx => {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }
            Self::Xlsx => {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
            Self::Pdf => "application/pdf",
            Self::Image => "image/png",
            Self::Latex => "application/x-tex",
            Self::XmlJats => "application/xml",
            Self::XmlUspto => "application/xml",
            Self::XmlXbrl => "application/xml",
            Self::JsonDocling => "application/json",
            Self::MetsGbs => "application/xml",
            Self::Audio => "audio/mpeg",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[serde(rename = "md")]
    Markdown,
    Json,
    Yaml,
    Html,
    Text,
    Csv,
    #[serde(rename = "doctags")]
    DocTags,
    #[serde(rename = "vtt")]
    Vtt,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Markdown => write!(f, "md"),
            Self::Json => write!(f, "json"),
            Self::Yaml => write!(f, "yaml"),
            Self::Html => write!(f, "html"),
            Self::Text => write!(f, "text"),
            Self::Csv => write!(f, "csv"),
            Self::DocTags => write!(f, "doctags"),
            Self::Vtt => write!(f, "vtt"),
        }
    }
}

impl OutputFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Markdown => "md",
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Html => "html",
            Self::Text => "txt",
            Self::Csv => "csv",
            Self::DocTags => "doctags",
            Self::Vtt => "vtt",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocItemLabel {
    Title,
    #[serde(rename = "section_header")]
    SectionHeader,
    Paragraph,
    Text,
    #[serde(rename = "list_item")]
    ListItem,
    Table,
    Picture,
    Formula,
    Code,
    Caption,
    #[serde(rename = "page_header")]
    PageHeader,
    #[serde(rename = "page_footer")]
    PageFooter,
    Footnote,
    #[serde(rename = "document_index")]
    DocumentIndex,
    Reference,
    Chart,
    #[serde(rename = "key_value_region")]
    KeyValueRegion,
    Form,
    #[serde(rename = "checkbox_selected")]
    CheckboxSelected,
    #[serde(rename = "checkbox_unselected")]
    CheckboxUnselected,
    #[serde(rename = "empty_value")]
    EmptyValue,
    Unspecified,
    #[serde(other)]
    Unknown,
}

impl Default for DocItemLabel {
    fn default() -> Self {
        Self::Text
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentLayer {
    Body,
    Furniture,
    Background,
    Invisible,
    Notes,
    #[serde(other)]
    Unknown,
}

impl Default for ContentLayer {
    fn default() -> Self {
        Self::Body
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupLabel {
    Unspecified,
    List,
    #[serde(rename = "ordered_list")]
    OrderedList,
    Chapter,
    Section,
    Sheet,
    Slide,
    #[serde(rename = "form_area")]
    FormArea,
    #[serde(rename = "key_value_area")]
    KeyValueArea,
    #[serde(rename = "comment_section")]
    CommentSection,
    Inline,
    #[serde(rename = "picture_area")]
    PictureArea,
    #[serde(other)]
    Unknown,
}

impl Default for GroupLabel {
    fn default() -> Self {
        Self::Unspecified
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageRefMode {
    #[default]
    Placeholder,
    Embedded,
    Referenced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversionStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "started")]
    Started,
    #[serde(rename = "failure")]
    Failure,
    #[serde(rename = "success")]
    Success,
    #[serde(rename = "partial_success")]
    PartialSuccess,
    #[serde(rename = "skipped")]
    Skipped,
    #[serde(other)]
    Unknown,
}
