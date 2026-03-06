use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use clap::Parser;

use docling::cli::{Cli, Commands};
use docling::converter::DocumentConverter;
use docling::export;
use docling::models::common::{ImageRefMode, InputFormat, OutputFormat};

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
            let output_formats: Vec<OutputFormat> =
                args.to.into_iter().map(|f| f.into()).collect();
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
    for fmt in output_formats {
        let content = export::export_document(&doc, fmt, Some(image_mode))?;
        let out_name = format!("{}.{}", stem, fmt.extension());
        let out_path = output_dir.join(&out_name);
        fs::write(&out_path, &content)?;
        log::info!("  Wrote: {}", out_path.display());
    }

    Ok(())
}

fn materialize_images(
    doc: &mut docling::models::document::DoclingDocument,
    output_dir: &Path,
) -> anyhow::Result<()> {
    use base64::Engine;

    let images_dir = output_dir.join(format!("{}_images", doc.name));
    let mut created_dir = false;

    for (i, pic) in doc.pictures.iter_mut().enumerate() {
        let img = match pic.image.as_mut() {
            Some(img) if img.uri.starts_with("data:") => img,
            _ => continue,
        };

        if !created_dir {
            fs::create_dir_all(&images_dir)?;
            created_dir = true;
        }

        let ext = match img.mimetype.as_str() {
            "image/jpeg" => "jpg",
            "image/png" => "png",
            "image/gif" => "gif",
            "image/bmp" => "bmp",
            "image/tiff" => "tif",
            "image/webp" => "webp",
            "image/x-emf" => "emf",
            "image/x-wmf" => "wmf",
            "image/svg+xml" => "svg",
            _ => "png",
        };

        let filename = format!("image_{}.{}", i, ext);
        let file_path = images_dir.join(&filename);

        if let Some(b64_start) = img.uri.find(";base64,") {
            let b64_data = &img.uri[b64_start + 8..];
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64_data) {
                fs::write(&file_path, &bytes)?;
                img.uri = format!("{}_images/{}", doc.name, filename);
                log::info!("  Saved image: {}", file_path.display());
            }
        }
    }

    Ok(())
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
        let stem = f
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
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
