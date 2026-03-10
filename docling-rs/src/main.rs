use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use clap::Parser;
use sha2::{Digest, Sha256};

use docling::cli::{Cli, Commands};
use docling::converter::DocumentConverter;
use docling::export;
use docling::models::common::{ImageRefMode, InputFormat, OutputFormat};

/// Minimum image dimensions - images smaller than this are filtered out
const MIN_IMAGE_WIDTH: u32 = 200;
const MIN_IMAGE_HEIGHT: u32 = 200;

/// Image compression settings (matching docsextract/compressor.py)
const MAX_IMAGE_DIMENSION: u32 = 1536;
const MAX_IMAGE_SIZE_KB: usize = 500;
const JPEG_QUALITY_START: u8 = 95;
const JPEG_QUALITY_MIN: u8 = 30;
const JPEG_QUALITY_STEP: u8 = 10;

/// Minimum edge score to consider an image as having actual content
/// Background/gradient images have very low edge scores (< 3.0)
/// Content images typically have edge scores > 5.0
const MIN_EDGE_SCORE: f32 = 3.0;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Convert(args) => {
            match args.verbose {
                0 => env_logger::Builder::new()
                    .filter_level(log::LevelFilter::Warn)
                    .init(),
                1 => env_logger::Builder::new()
                    .filter_level(log::LevelFilter::Info)
                    .init(),
                _ => env_logger::Builder::new()
                    .filter_level(log::LevelFilter::Debug)
                    .init(),
            }

            let converter = DocumentConverter::new();
            let output_dir = &args.output;
            fs::create_dir_all(output_dir)?;

            let input_format = args.from.map(|f| f.into());
            let output_formats: Vec<OutputFormat> = args.to.into_iter().map(|f| f.into()).collect();
            let image_mode = args
                .image_export_mode
                .map(|m| m.into())
                .unwrap_or(ImageRefMode::Placeholder);
            let timeout = args.document_timeout.map(Duration::from_secs);

            let source_files = resolve_sources(&args.source)?;

            let opts = ConvertOptions {
                input_format: input_format.as_ref(),
                output_formats: &output_formats,
                output_dir,
                image_mode,
                timeout,
                abort_on_error: args.abort_on_error,
            };

            let pool_size = (args.num_threads as usize).max(1);
            let errors = if pool_size > 1 && source_files.len() > 1 {
                convert_parallel(&converter, &source_files, &opts, pool_size)
            } else {
                convert_sequential(&converter, &source_files, &opts)
            };

            errors?;
        }
    }

    Ok(())
}

fn convert_one(
    converter: &DocumentConverter,
    source: &Path,
    input_format: Option<&InputFormat>,
    output_formats: &[OutputFormat],
    output_dir: &Path,
    image_mode: ImageRefMode,
    timeout: Option<Duration>,
) -> anyhow::Result<()> {
    log::info!("Converting: {}", source.display());

    let mut doc = if let Some(dur) = timeout {
        let source = source.to_path_buf();
        let input_format = input_format.cloned();
        run_with_timeout(dur, move || {
            let converter = DocumentConverter::new();
            converter.convert(&source, input_format.as_ref())
        })?
    } else {
        converter.convert(source, input_format)?
    };

    if image_mode == ImageRefMode::Referenced {
        materialize_images(&mut doc, output_dir)?;
    }

    let stem = &doc.name;
    
    // Check if this is an XLSX file - if so, export each sheet as separate CSV
    let is_xlsx = source
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("xlsx") || e.eq_ignore_ascii_case("xls"))
        .unwrap_or(false);
    
    if is_xlsx {
        // Export each sheet as a separate CSV file
        export_xlsx_sheets_as_csv(source, output_dir, stem)?;
    }
    
    for fmt in output_formats {
        let content = export::export_document(&doc, fmt, Some(image_mode))?;
        let out_name = format!("{}.{}", stem, fmt.extension());
        let out_path = output_dir.join(&out_name);
        fs::write(&out_path, &content)?;
        log::info!("  Wrote: {}", out_path.display());
    }

    Ok(())
}

/// Calculate edge score for an image to detect backgrounds/gradients.
/// Higher scores indicate more visual complexity (actual content).
/// Low scores indicate solid colors or gradients (backgrounds).
fn calculate_edge_score(img: &image::DynamicImage) -> f32 {
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    
    if width < 2 || height < 2 {
        return 0.0;
    }
    
    let pixels = rgb.as_raw();
    let mut total_diff: f64 = 0.0;
    let mut count: u64 = 0;
    
    // Sample pixels (every 10th pixel for performance)
    let step = 10;
    
    // Horizontal differences
    for y in (0..height).step_by(step) {
        for x in (0..width - 1).step_by(step) {
            let idx1 = (y * width + x) * 3;
            let idx2 = (y * width + x + 1) * 3;
            
            let dr = (pixels[idx1] as i32 - pixels[idx2] as i32).abs();
            let dg = (pixels[idx1 + 1] as i32 - pixels[idx2 + 1] as i32).abs();
            let db = (pixels[idx1 + 2] as i32 - pixels[idx2 + 2] as i32).abs();
            
            total_diff += (dr + dg + db) as f64;
            count += 1;
        }
    }
    
    // Vertical differences
    for y in (0..height - 1).step_by(step) {
        for x in (0..width).step_by(step) {
            let idx1 = (y * width + x) * 3;
            let idx2 = ((y + 1) * width + x) * 3;
            
            let dr = (pixels[idx1] as i32 - pixels[idx2] as i32).abs();
            let dg = (pixels[idx1 + 1] as i32 - pixels[idx2 + 1] as i32).abs();
            let db = (pixels[idx1 + 2] as i32 - pixels[idx2 + 2] as i32).abs();
            
            total_diff += (dr + dg + db) as f64;
            count += 1;
        }
    }
    
    if count == 0 {
        return 0.0;
    }
    
    (total_diff / count as f64) as f32
}

fn materialize_images(
    doc: &mut docling::models::document::DoclingDocument,
    output_dir: &Path,
) -> anyhow::Result<()> {
    use base64::Engine;

    let images_dir = output_dir.join(format!("{}_images", doc.name));
    let mut created_dir = false;
    
    // Track seen image hashes for deduplication
    let mut seen_hashes: HashMap<String, usize> = HashMap::new();
    // Map original index to new index (for deduplication)
    let mut index_map: HashMap<usize, usize> = HashMap::new();
    let mut kept_count: usize = 0;

    for (i, pic) in doc.pictures.iter_mut().enumerate() {
        let img = match pic.image.as_mut() {
            Some(img) if img.uri.starts_with("data:") => img,
            _ => continue,
        };

        let bytes = if let Some(b64_start) = img.uri.find(";base64,") {
            let b64_data = &img.uri[b64_start + 8..];
            match base64::engine::general_purpose::STANDARD.decode(b64_data) {
                Ok(b) => b,
                Err(_) => continue,
            }
        } else {
            continue;
        };

        // Compute content hash for deduplication
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());
        let hash_prefix = &hash[..16];

        // Check for duplicate
        if let Some(&existing_idx) = seen_hashes.get(hash_prefix) {
            index_map.insert(i, existing_idx);
            log::info!("  Skipping duplicate image {} (same as image {})", i, existing_idx);
            // Clear the URI to mark as duplicate
            img.uri = format!("{}_images/image_{:02}.jpg", doc.name, existing_idx);
            continue;
        }

        // Try to load the image to check dimensions
        let dynamic_img = match image::load_from_memory(&bytes) {
            Ok(img) => img,
            Err(_) => {
                // Can't decode image, skip filtering but keep it
                log::warn!("  Could not decode image {}, keeping without filtering", i);
                if !created_dir {
                    fs::create_dir_all(&images_dir)?;
                    created_dir = true;
                }
                let filename = format!("image_{:02}.jpg", kept_count);
                let file_path = images_dir.join(&filename);
                fs::write(&file_path, &bytes)?;
                img.uri = format!("{}_images/{}", doc.name, filename);
                seen_hashes.insert(hash_prefix.to_string(), kept_count);
                index_map.insert(i, kept_count);
                kept_count += 1;
                continue;
            }
        };

        let (width, height) = (dynamic_img.width(), dynamic_img.height());

        // Filter out small images (likely icons/logos)
        // Use OR to be stricter - filter if EITHER dimension is too small
        if width < MIN_IMAGE_WIDTH || height < MIN_IMAGE_HEIGHT {
            log::info!("  Filtering small image {} ({}x{})", i, width, height);
            img.uri = format!("[IMAGE OMITTED: too small ({}x{}px)]", width, height);
            continue;
        }

        // Filter out background/gradient images (low visual complexity)
        let edge_score = calculate_edge_score(&dynamic_img);
        if edge_score < MIN_EDGE_SCORE {
            log::info!("  Filtering background image {} (edge_score={:.1})", i, edge_score);
            img.uri = format!("[IMAGE OMITTED: background/gradient (edge_score={:.1})]", edge_score);
            continue;
        }

        if !created_dir {
            fs::create_dir_all(&images_dir)?;
            created_dir = true;
        }

        // Resize if exceeds max dimension
        let resized_img = if width > MAX_IMAGE_DIMENSION || height > MAX_IMAGE_DIMENSION {
            let ratio = (MAX_IMAGE_DIMENSION as f32 / width as f32)
                .min(MAX_IMAGE_DIMENSION as f32 / height as f32);
            let new_width = (width as f32 * ratio) as u32;
            let new_height = (height as f32 * ratio) as u32;
            log::info!("  Resizing image {} from {}x{} to {}x{}", i, width, height, new_width, new_height);
            dynamic_img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
        } else {
            dynamic_img
        };

        let rgb_img = resized_img.to_rgb8();
        let (final_width, final_height) = (rgb_img.width(), rgb_img.height());

        // Compress with decreasing quality until size target is met
        let max_size_bytes = MAX_IMAGE_SIZE_KB * 1024;
        let mut quality = JPEG_QUALITY_START;
        let mut jpeg_bytes: Vec<u8>;

        loop {
            jpeg_bytes = Vec::new();
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                Cursor::new(&mut jpeg_bytes),
                quality,
            );
            encoder.encode_image(&rgb_img)?;

            if jpeg_bytes.len() <= max_size_bytes || quality <= JPEG_QUALITY_MIN {
                break;
            }
            quality = quality.saturating_sub(JPEG_QUALITY_STEP);
        }

        let filename = format!("image_{:02}.jpg", kept_count);
        let file_path = images_dir.join(&filename);
        
        fs::write(&file_path, &jpeg_bytes)?;
        img.uri = format!("{}_images/{}", doc.name, filename);
        img.mimetype = "image/jpeg".to_string();
        
        seen_hashes.insert(hash_prefix.to_string(), kept_count);
        index_map.insert(i, kept_count);
        log::info!(
            "  Saved image {}: {} ({}x{}, {}KB, q={})",
            kept_count, file_path.display(), final_width, final_height,
            jpeg_bytes.len() / 1024, quality
        );
        kept_count += 1;
    }

    log::info!("  Kept {} images after filtering/deduplication", kept_count);
    Ok(())
}

/// Export each sheet in an XLSX file as a separate CSV file.
/// Files are named: {stem}_{sheet_name}.csv
fn export_xlsx_sheets_as_csv(
    xlsx_path: &Path,
    output_dir: &Path,
    stem: &str,
) -> anyhow::Result<()> {
    use calamine::{open_workbook_auto, Reader};

    let mut workbook = open_workbook_auto(xlsx_path)?;
    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();

    for sheet_name in &sheet_names {
        let range = match workbook.worksheet_range(sheet_name) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let (height, width) = range.get_size();
        if height == 0 || width == 0 {
            continue;
        }

        // Sanitize sheet name for filename
        let safe_name: String = sheet_name
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect();

        let csv_filename = format!("{}_{}.csv", stem, safe_name);
        let csv_path = output_dir.join(&csv_filename);

        let mut csv_content = String::new();
        for row in range.rows() {
            let cells: Vec<String> = row.iter().map(|cell| cell_to_csv_string(cell)).collect();
            csv_content.push_str(&cells.join(","));
            csv_content.push('\n');
        }

        fs::write(&csv_path, &csv_content)?;
        log::info!("  Wrote CSV sheet: {}", csv_path.display());
    }

    Ok(())
}

/// Convert a cell value to CSV-safe string
fn cell_to_csv_string(cell: &calamine::Data) -> String {
    use calamine::Data;
    
    let value = match cell {
        Data::Int(i) => i.to_string(),
        Data::Float(f) => {
            if f.is_nan() || f.is_infinite() {
                return String::new();
            }
            if *f == f.floor() && f.abs() < 1e15 {
                format!("{}", *f as i64)
            } else {
                f.to_string()
            }
        }
        Data::String(s) => s.clone(),
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => match dt.as_datetime() {
            Some(naive) => format!("{}", naive.format("%Y-%m-%d %H:%M:%S")),
            None => format!("{}", dt),
        },
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(_) => String::new(),
        Data::Empty => String::new(),
    };

    // RFC 4180 CSV escaping
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        let escaped = value.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        value
    }
}

fn run_with_timeout<F, T>(dur: Duration, f: F) -> anyhow::Result<T>
where
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = f();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(dur) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            anyhow::bail!("Conversion exceeded timeout of {}s", dur.as_secs())
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            anyhow::bail!("Conversion thread panicked")
        }
    }
}

struct ConvertOptions<'a> {
    input_format: Option<&'a InputFormat>,
    output_formats: &'a [OutputFormat],
    output_dir: &'a Path,
    image_mode: ImageRefMode,
    timeout: Option<Duration>,
    abort_on_error: bool,
}

fn convert_sequential(
    converter: &DocumentConverter,
    source_files: &[std::path::PathBuf],
    opts: &ConvertOptions<'_>,
) -> anyhow::Result<()> {
    let mut last_error: Option<anyhow::Error> = None;
    let mut success_count = 0usize;
    for source in source_files {
        if let Err(e) = convert_one(
            converter,
            source,
            opts.input_format,
            opts.output_formats,
            opts.output_dir,
            opts.image_mode,
            opts.timeout,
        ) {
            log::error!("Failed to convert {}: {}", source.display(), e);
            if opts.abort_on_error {
                return Err(e);
            }
            last_error = Some(e);
        } else {
            success_count += 1;
        }
    }
    if success_count == 0 {
        if let Some(e) = last_error {
            return Err(e);
        }
    }
    Ok(())
}

fn convert_parallel(
    converter: &DocumentConverter,
    source_files: &[std::path::PathBuf],
    opts: &ConvertOptions<'_>,
    pool_size: usize,
) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let should_abort = Arc::new(AtomicBool::new(false));

    std::thread::scope(|scope| {
        let chunks: Vec<&[std::path::PathBuf]> = source_files
            .chunks(source_files.len().div_ceil(pool_size))
            .collect();

        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let should_abort = Arc::clone(&should_abort);
                scope.spawn(move || -> anyhow::Result<()> {
                    for source in chunk {
                        if opts.abort_on_error && should_abort.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        if let Err(e) = convert_one(
                            converter,
                            source,
                            opts.input_format,
                            opts.output_formats,
                            opts.output_dir,
                            opts.image_mode,
                            opts.timeout,
                        ) {
                            log::error!("Failed to convert {}: {}", source.display(), e);
                            if opts.abort_on_error {
                                should_abort.store(true, Ordering::Relaxed);
                                return Err(e);
                            }
                        }
                    }
                    Ok(())
                })
            })
            .collect();

        let mut first_error = None;
        for handle in handles {
            if let Err(e) = handle.join().expect("thread panicked") {
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }

        match first_error {
            Some(e) if opts.abort_on_error => Err(e),
            _ => Ok(()),
        }
    })
}

fn resolve_sources(sources: &[std::path::PathBuf]) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    for source in sources {
        if source.is_dir() {
            let mut visited = HashSet::new();
            walkdir(source, &mut files, &mut visited)?;
        } else if source.is_file() {
            files.push(source.clone());
        } else {
            anyhow::bail!("Source not found: {}", source.display());
        }
    }
    files.sort();
    warn_duplicate_stems(&files);
    Ok(files)
}

fn warn_duplicate_stems(files: &[std::path::PathBuf]) {
    let mut seen = HashSet::new();
    for f in files {
        let stem = f.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
        if !seen.insert(stem.to_string()) {
            log::warn!(
                "Multiple source files share the stem '{}' — output files may overwrite each other. \
                 Consider converting them separately or renaming: {}",
                stem,
                f.display()
            );
        }
    }
}

fn walkdir(
    dir: &Path,
    files: &mut Vec<std::path::PathBuf>,
    visited: &mut HashSet<std::path::PathBuf>,
) -> anyhow::Result<()> {
    let canonical = dir.canonicalize()?;
    if !visited.insert(canonical) {
        log::warn!(
            "Skipping already-visited directory (symlink cycle?): {}",
            dir.display()
        );
        return Ok(());
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with('.'))
            .unwrap_or(false)
        {
            continue;
        }

        if path.is_dir() {
            walkdir(&path, files, visited)?;
        } else if path.is_file() && InputFormat::from_extension(&path).is_some() {
            files.push(path);
        }
    }
    Ok(())
}
