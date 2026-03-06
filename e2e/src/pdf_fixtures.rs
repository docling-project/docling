use std::io::Write;
use std::path::Path;
use std::sync::Once;

static GENERATE: Once = Once::new();

/// Ensure all PDF fixtures exist under `tests/data/pdf/`.
/// Safe to call multiple times — generation happens exactly once per process.
pub fn ensure_pdf_fixtures(test_data_dir: &Path) {
    let pdf_dir = test_data_dir.join("pdf");
    GENERATE.call_once(|| {
        std::fs::create_dir_all(&pdf_dir).expect("create pdf dir");
        generate_single_page(&pdf_dir.join("single_page.pdf"));
        generate_multi_page(&pdf_dir.join("multi_page_generated.pdf"));
        generate_headings(&pdf_dir.join("headings_and_text.pdf"));
        generate_empty(&pdf_dir.join("empty_page.pdf"));
    });
}

// ---------------------------------------------------------------------------
// Minimal PDF builder — constructs valid PDFs with Type1 fonts (no deps)
// ---------------------------------------------------------------------------

struct PdfBuilder {
    objects: Vec<(u32, String)>,
    next_id: u32,
}

impl PdfBuilder {
    fn new() -> Self {
        Self {
            objects: Vec::new(),
            next_id: 1,
        }
    }

    fn alloc(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn add(&mut self, id: u32, body: String) {
        self.objects.push((id, body));
    }

    fn build(self, catalog_id: u32) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");

        let mut offsets: Vec<(u32, usize)> = Vec::new();

        for (id, body) in &self.objects {
            offsets.push((*id, buf.len()));
            write!(buf, "{id} 0 obj\n{body}\nendobj\n").unwrap();
        }

        let xref_offset = buf.len();
        let max_id = self.next_id;
        write!(buf, "xref\n0 {max_id}\n").unwrap();
        writeln!(buf, "0000000000 65535 f ").unwrap();

        let mut sorted = offsets.clone();
        sorted.sort_by_key(|(id, _)| *id);
        for (_, offset) in &sorted {
            writeln!(buf, "{offset:010} 00000 n ").unwrap();
        }

        write!(
            buf,
            "trailer\n<< /Size {max_id} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        )
        .unwrap();
        buf
    }
}

fn escape_pdf_string(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '(' => out.push_str("\\("),
            ')' => out.push_str("\\)"),
            '\\' => out.push_str("\\\\"),
            c if c.is_ascii() => out.push(c),
            _ => out.push(' '),
        }
    }
    out
}

fn make_content_stream(texts: &[(f64, f64, f64, &str)]) -> String {
    let mut stream = String::new();
    for &(x, y, size, text) in texts {
        let escaped = escape_pdf_string(text);
        stream.push_str(&format!("BT /F1 {size} Tf {x} {y} Td ({escaped}) Tj ET\n"));
    }
    stream
}

fn build_page(
    builder: &mut PdfBuilder,
    parent_id: u32,
    font_id: u32,
    width: f64,
    height: f64,
    texts: &[(f64, f64, f64, &str)],
) -> u32 {
    let page_id = builder.alloc();
    let content_id = builder.alloc();

    let stream = make_content_stream(texts);
    let stream_len = stream.len();
    builder.add(
        content_id,
        format!("<< /Length {stream_len} >>\nstream\n{stream}endstream"),
    );

    builder.add(
        page_id,
        format!(
            concat!(
                "<< /Type /Page /Parent {parent} 0 R ",
                "/MediaBox [0 0 {w} {h}] ",
                "/Contents {contents} 0 R ",
                "/Resources << /Font << /F1 {font} 0 R >> >> >>"
            ),
            parent = parent_id,
            w = width,
            h = height,
            contents = content_id,
            font = font_id,
        ),
    );

    page_id
}

fn generate_single_page(path: &Path) {
    let mut b = PdfBuilder::new();
    let catalog_id = b.alloc();
    let pages_id = b.alloc();
    let font_id = b.alloc();

    b.add(
        font_id,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".into(),
    );

    let texts = [
        (72.0, 700.0, 24.0, "Test Document Title"),
        (
            72.0,
            660.0,
            12.0,
            "This is the first paragraph of the test document. It contains some text that should be extracted correctly by the PDF backend.",
        ),
        (
            72.0,
            620.0,
            12.0,
            "This is the second paragraph. It provides additional content for testing paragraph splitting and text extraction.",
        ),
    ];

    let page_id = build_page(&mut b, pages_id, font_id, 612.0, 792.0, &texts);

    b.add(
        pages_id,
        format!("<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>"),
    );
    b.add(
        catalog_id,
        format!("<< /Type /Catalog /Pages {pages_id} 0 R >>"),
    );

    std::fs::write(path, b.build(catalog_id)).expect("write single_page.pdf");
}

fn generate_multi_page(path: &Path) {
    let mut b = PdfBuilder::new();
    let catalog_id = b.alloc();
    let pages_id = b.alloc();
    let font_id = b.alloc();

    b.add(
        font_id,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".into(),
    );

    let p1 = build_page(
        &mut b,
        pages_id,
        font_id,
        612.0,
        792.0,
        &[
            (72.0, 700.0, 20.0, "1 Introduction"),
            (
                72.0,
                660.0,
                12.0,
                "This is the first page of a multi-page document. It introduces the main topic.",
            ),
        ],
    );

    let p2 = build_page(
        &mut b,
        pages_id,
        font_id,
        612.0,
        792.0,
        &[
            (72.0, 700.0, 16.0, "2 Methods"),
            (
                72.0,
                660.0,
                12.0,
                "The second page describes the methods used in this study.",
            ),
        ],
    );

    let p3 = build_page(
        &mut b,
        pages_id,
        font_id,
        612.0,
        792.0,
        &[
            (72.0, 700.0, 16.0, "3 Conclusion"),
            (
                72.0,
                660.0,
                12.0,
                "The third page wraps up with conclusions.",
            ),
        ],
    );

    b.add(
        pages_id,
        format!("<< /Type /Pages /Kids [{p1} 0 R {p2} 0 R {p3} 0 R] /Count 3 >>"),
    );
    b.add(
        catalog_id,
        format!("<< /Type /Catalog /Pages {pages_id} 0 R >>"),
    );

    std::fs::write(path, b.build(catalog_id)).expect("write multi_page_generated.pdf");
}

fn generate_headings(path: &Path) {
    let mut b = PdfBuilder::new();
    let catalog_id = b.alloc();
    let pages_id = b.alloc();
    let font_id = b.alloc();

    b.add(
        font_id,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".into(),
    );

    let page_id = build_page(
        &mut b,
        pages_id,
        font_id,
        612.0,
        792.0,
        &[
            (72.0, 720.0, 28.0, "Document Title"),
            (72.0, 680.0, 12.0, "A paragraph under the title with enough text to be clearly body content in any analysis."),
            (72.0, 640.0, 20.0, "1.1 First Section"),
            (72.0, 600.0, 12.0, "Content of the first section that discusses important details at length."),
            (72.0, 560.0, 20.0, "1.2 Second Section"),
            (72.0, 520.0, 12.0, "Content of the second section with more details to analyze."),
        ],
    );

    b.add(
        pages_id,
        format!("<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>"),
    );
    b.add(
        catalog_id,
        format!("<< /Type /Catalog /Pages {pages_id} 0 R >>"),
    );

    std::fs::write(path, b.build(catalog_id)).expect("write headings_and_text.pdf");
}

fn generate_empty(path: &Path) {
    let mut b = PdfBuilder::new();
    let catalog_id = b.alloc();
    let pages_id = b.alloc();

    let page_id = b.alloc();
    let content_id = b.alloc();

    let stream = "";
    b.add(
        content_id,
        format!("<< /Length 0 >>\nstream\n{stream}endstream"),
    );

    b.add(
        page_id,
        format!(
            "<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R /Resources << >> >>"
        ),
    );

    b.add(
        pages_id,
        format!("<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>"),
    );
    b.add(
        catalog_id,
        format!("<< /Type /Catalog /Pages {pages_id} 0 R >>"),
    );

    std::fs::write(path, b.build(catalog_id)).expect("write empty_page.pdf");
}
