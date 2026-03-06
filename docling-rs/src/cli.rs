use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, ValueEnum)]
pub enum CliInputFormat {
    Csv,
    Md,
    Html,
    #[value(name = "asciidoc")]
    AsciiDoc,
    Vtt,
    Docx,
    Pptx,
    Xlsx,
    Pdf,
    Image,
    Latex,
    #[value(name = "xml_jats")]
    XmlJats,
    #[value(name = "xml_uspto")]
    XmlUspto,
    #[value(name = "xml_xbrl")]
    XmlXbrl,
    #[value(name = "json_docling")]
    JsonDocling,
    #[value(name = "mets_gbs")]
    MetsGbs,
}

impl From<CliInputFormat> for crate::models::common::InputFormat {
    fn from(f: CliInputFormat) -> Self {
        match f {
            CliInputFormat::Csv => Self::Csv,
            CliInputFormat::Md => Self::Md,
            CliInputFormat::Html => Self::Html,
            CliInputFormat::AsciiDoc => Self::AsciiDoc,
            CliInputFormat::Vtt => Self::Vtt,
            CliInputFormat::Docx => Self::Docx,
            CliInputFormat::Pptx => Self::Pptx,
            CliInputFormat::Xlsx => Self::Xlsx,
            CliInputFormat::Pdf => Self::Pdf,
            CliInputFormat::Image => Self::Image,
            CliInputFormat::Latex => Self::Latex,
            CliInputFormat::XmlJats => Self::XmlJats,
            CliInputFormat::XmlUspto => Self::XmlUspto,
            CliInputFormat::XmlXbrl => Self::XmlXbrl,
            CliInputFormat::JsonDocling => Self::JsonDocling,
            CliInputFormat::MetsGbs => Self::MetsGbs,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum CliOutputFormat {
    Md,
    Json,
    Yaml,
    Html,
    Text,
    Csv,
    #[value(name = "doctags")]
    DocTags,
    Vtt,
}

impl From<CliOutputFormat> for crate::models::common::OutputFormat {
    fn from(f: CliOutputFormat) -> Self {
        match f {
            CliOutputFormat::Md => Self::Markdown,
            CliOutputFormat::Json => Self::Json,
            CliOutputFormat::Yaml => Self::Yaml,
            CliOutputFormat::Html => Self::Html,
            CliOutputFormat::Text => Self::Text,
            CliOutputFormat::Csv => Self::Csv,
            CliOutputFormat::DocTags => Self::DocTags,
            CliOutputFormat::Vtt => Self::Vtt,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum CliImageRefMode {
    Placeholder,
    Embedded,
    Referenced,
}

impl From<CliImageRefMode> for crate::models::common::ImageRefMode {
    fn from(m: CliImageRefMode) -> Self {
        match m {
            CliImageRefMode::Placeholder => Self::Placeholder,
            CliImageRefMode::Embedded => Self::Embedded,
            CliImageRefMode::Referenced => Self::Referenced,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "docling-rs", version, about = "Document conversion tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Convert documents to various output formats
    Convert(ConvertArgs),
}

#[derive(clap::Args, Debug)]
pub struct ConvertArgs {
    /// Input files or directories
    #[arg(required = true)]
    pub source: Vec<PathBuf>,

    /// Input format (auto-detected if not specified)
    #[arg(short = 'f', long = "from")]
    pub from: Option<CliInputFormat>,

    /// Output format(s)
    #[arg(long = "to", default_value = "md")]
    pub to: Vec<CliOutputFormat>,

    /// Output directory
    #[arg(short = 'o', long = "output", default_value = ".")]
    pub output: PathBuf,

    /// Verbosity level (-v for info, -vv for debug)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Abort on first error
    #[arg(long = "abort-on-error")]
    pub abort_on_error: bool,

    /// Per-document timeout in seconds
    #[arg(long = "document-timeout")]
    pub document_timeout: Option<u64>,

    /// Number of threads
    #[arg(long = "num-threads", default_value = "4")]
    pub num_threads: u32,

    /// Image export mode
    #[arg(long = "image-export-mode")]
    pub image_export_mode: Option<CliImageRefMode>,
}
