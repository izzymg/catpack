use std::error::Error;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use catpack::{CompressionType, PackageBuilder, PackageChunk, PackageHandle, WriteResult};

use clap::{Arg, Command};

const EXTENSION: &str = "cpkg";

struct Config {
    quiet: bool,
    chunk_count: usize,
    compression_type: CompressionType,
}

fn unpack(
    input_pkg: &path::Path,
    output_path: &path::Path,
) -> Result<(Duration, usize), Box<dyn Error>> {
    let mut bytes_written = 0;
    let time_t = Instant::now();
    let file = fs::File::open(input_pkg)?;
    let mut package = PackageHandle::from_file(file)?;
    let entries = package.iter().cloned().collect::<Vec<_>>();

    if !output_path.exists() {
        fs::create_dir(output_path)?;
    }

    let mut buffer = Vec::new();

    for entry in entries.iter() {
        buffer.clear();
        match package.read_by_id(entry, &mut buffer) {
            Some(Ok(())) => {}
            Some(Err(e)) => {
                return Err(format!("Failed to read file '{entry}' from package: {e}").into());
            }
            None => return Err(format!("File '{entry}' not found in package").into()),
        }

        let path = path::Path::new(output_path).join(entry);
        let parent = path
            .parent()
            .ok_or_else(|| format!("Invalid path for file '{entry}'"))?;
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(path)?;
        file.write_all(&buffer)?;
        bytes_written += buffer.len();
    }

    Ok((time_t.elapsed(), bytes_written))
}

fn flush_chunks(
    package: &mut PackageBuilder,
    output_file: &mut File,
    chunks: Vec<PackageChunk>,
) -> io::Result<WriteResult> {
    let chunk_size = chunks
        .iter()
        .fold(0, |ac, chunk: &PackageChunk| ac + chunk.raw_bytes.len());
    log::info!(
        "writing chunk to disk: {} files, {chunk_size} bytes",
        chunks.len()
    );
    package.write_chunks(output_file, chunks)
}

struct PackResult {
    time_taken: Duration,
    bytes_written: usize,
    compressed_data_size: usize,
    uncompressed_data_size: usize,
}

fn pack(
    config: &Config,
    input_dir: &path::Path,
    mut output_path: path::PathBuf,
) -> Result<PackResult, Box<dyn std::error::Error>> {
    if !input_dir.exists() {
        return Err(format!("target directory does not exist: {input_dir:#?}").into());
    }
    let mut bytes_written = 0;
    let mut compressed_data_size = 0;
    let mut uncompressed_data_size = 0;

    if output_path.extension().is_none() {
        output_path = output_path.with_extension(EXTENSION);
    }
    let mut output_file = fs::File::create(output_path)?;
    let input_dir_name = input_dir
        .file_stem()
        .ok_or("Input directory must have a valid name")?;
    let start_time = Instant::now();
    // For now we'll just support 1 level of depth
    let mut package = PackageBuilder::new();
    bytes_written += package.write_header(&mut output_file)?;

    let files_to_process = input_dir
        .read_dir()?
        .filter_map(|file| {
            let file_path = file.expect("failed to read file in directory").path();
            if file_path.is_dir() {
                return None;
            }
            Some(file_path)
        })
        .collect::<Vec<_>>();

    let total_to_process = files_to_process.len();

    let mut chunks = vec![];
    for (i, file_path) in files_to_process.iter().enumerate() {
        if chunks.len() >= config.chunk_count {
            let flush = std::mem::take(&mut chunks);
            let result = flush_chunks(&mut package, &mut output_file, flush)?;
            uncompressed_data_size += result.total_uncompressed_size;
            compressed_data_size += result.total_compressed_size;
            bytes_written += result.total_compressed_size;
        }

        // Handle UTF-8 conversion properly
        let file_id = match file_path.file_name() {
            Some(stem) => {
                let dir_name = input_dir_name.to_str().ok_or_else(|| {
                    format!("Directory name contains invalid UTF-8: {input_dir_name:?}")
                })?;
                let file_name = stem
                    .to_str()
                    .ok_or_else(|| format!("File name contains invalid UTF-8: {stem:?}"))?;
                format!("{dir_name}/{file_name}")
            }
            None => {
                continue;
            }
        };
        log::info!("reading: {file_id} ({i}/{total_to_process})");
        let chunk = match fs::read(file_path) {
            Ok(data) => PackageChunk::new(file_id, data, config.compression_type),
            Err(e) => {
                log::warn!("failed to read: {file_path:?}, skipping: {e}");
                continue;
            }
        };
        chunks.push(chunk);
    }
    let result = flush_chunks(&mut package, &mut output_file, chunks)?;
    uncompressed_data_size += result.total_uncompressed_size;
    compressed_data_size += result.total_compressed_size;
    bytes_written += result.total_compressed_size;
    bytes_written += package.write_toc(&mut output_file)?;
    let time_taken = start_time.elapsed();

    Ok(PackResult {
        time_taken,
        bytes_written,
        uncompressed_data_size,
        compressed_data_size,
    })
}

fn main() -> ExitCode {
    let matches = Command::new("packer")
        .about("Packs or unpacks files")
        .subcommand_required(true) // Must pick a subcommand
        .arg_required_else_help(true) // Show help if missing args
        .arg(Arg::new("quiet").long("quiet").short('q').help("Disables stdout").action(clap::ArgAction::SetTrue))
        .arg(Arg::new("compress").long("compress").short('c').help("Uses lz4 de/compression on all files").action(clap::ArgAction::SetTrue))
        .arg(Arg::new("chunk-count")
        .help("Specifies the maximum number of files to load into memory at once when packaging a directory")
            .long("chunk-count")
            .default_value("2")
            .value_name("CHUNK COUNT")
            .value_parser(clap::value_parser!(usize)))
        .subcommand(
            Command::new("pack")
                .about("Pack files into an archive")
                .arg(
                    Arg::new("input")
                        .help("Path to the input file or directory")
                        .required(true)
                        .value_name("INPUT")
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                )
                .arg(
                    Arg::new("output")
                        .help("Path to the output archive")
                        .required(true)
                        .value_name("OUTPUT")
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                ),
        )
        .subcommand(
            Command::new("unpack")
                .about("Unpack files from an archive")
                .arg(
                    Arg::new("input")
                        .help("Path to the input archive")
                        .required(true)
                        .value_name("INPUT")
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                )
                .arg(
                    Arg::new("output")
                        .help("Path to the output directory")
                        .required(true)
                        .value_name("OUTPUT")
                        .value_parser(clap::value_parser!(std::path::PathBuf)),
                ),
        )
        .get_matches();

    let mut cfg = Config {
        chunk_count: 2,
        quiet: false,
        compression_type: CompressionType::None,
    };

    if matches.get_flag("compress") {
        cfg.compression_type = CompressionType::LZ4;
    }

    if matches.get_flag("quiet") {
        cfg.quiet = true;
    }
    cfg.chunk_count = *matches.get_one("chunk-count").unwrap();
    let level = if cfg.quiet {
        log::LevelFilter::Error
    } else {
        log::LevelFilter::Info
    };
    env_logger::Builder::new().filter_level(level).init();

    match matches.subcommand() {
        Some(("pack", sub_m)) => {
            let input: &std::path::PathBuf = sub_m.get_one("input").unwrap();
            let output: &std::path::PathBuf = sub_m.get_one("output").unwrap();

            match pack(&cfg, input, output.clone()) {
                Ok(result) => {
                    log::info!(
                        "done packaging: {output:#?} ({} bytes)",
                        result.bytes_written
                    );
                    if !matches!(cfg.compression_type, CompressionType::None) {
                        let savings_frac = (result.compressed_data_size as f32
                            / result.uncompressed_data_size as f32)
                            * 100.;
                        let savings_bytes =
                            result.uncompressed_data_size - result.compressed_data_size;
                        log::info!(
                            "\t uncompressed data size: {} bytes",
                            result.uncompressed_data_size
                        );
                        log::info!(
                            "\t compressed data size: {} bytes",
                            result.compressed_data_size
                        );
                        log::info!("\t reduction: {savings_bytes} bytes ({savings_frac:.2}%)")
                    } else {
                        log::info!(
                            "\t total data size: {} bytes (no compression)",
                            result.compressed_data_size
                        );
                    }
                    log::info!("\t took: {:#?}", result.time_taken);
                }
                Err(e) => {
                    eprintln!("error packing: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        Some(("unpack", sub_m)) => {
            let input: &std::path::PathBuf = sub_m.get_one("input").unwrap();
            let output: &std::path::PathBuf = sub_m.get_one("output").unwrap();
            match unpack(input, output) {
                Ok((time, written)) => {
                    log::info!("done unpacking to {output:#?}");
                    log::info!("\twrote {written} bytes in {}ms", time.as_millis())
                }
                Err(e) => {
                    eprintln!("error unpacking: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        _ => unreachable!("Subcommand required by clap"),
    }

    ExitCode::SUCCESS
}
