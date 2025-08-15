use std::fs::{self, File};
use std::io::{self, Write};
use std::path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use catpack::{PackageBuilder, PackageChunk, PackageHandle};

use clap::{Arg, Command};

const EXTENSION: &str = "cpkg";

struct Config {
    quiet: bool,
    chunk_count: usize,
}

fn unpack(input_pkg: &path::Path, output_path: &path::Path) -> io::Result<(Duration, usize)> {
    let mut bytes_written = 0;
    let time_t = Instant::now();
    let file = fs::File::open(input_pkg)?;
    let mut package = PackageHandle::from_file(file).expect("pkg did not open");
    let entries = package.iter().cloned().collect::<Vec<_>>();

    if !output_path.exists() {
        fs::create_dir(output_path)?;
    }

    let mut buffer = Vec::new();

    for entry in entries.iter() {
        buffer.clear();
        let needed_size = package.get_data_size(entry).expect("this should exist");
        if buffer.len() < needed_size {
            buffer.resize(needed_size, 0u8);
        }
        package
            .read_by_id(entry, &mut buffer)
            .expect("this also should exist")?;

        let path = path::Path::new(output_path).join(entry);
        let parent = path.parent().unwrap();
        if !parent.exists() {
            fs::create_dir(path.parent().unwrap())?;
        }
        let mut file = fs::File::create(path)?;
        file.write_all(&buffer)?;
        bytes_written += needed_size;
    }

    Ok((time_t.elapsed(), bytes_written))
}

fn flush_chunks(
    package: &mut PackageBuilder,
    output_file: &mut File,
    chunks: Vec<PackageChunk>,
) -> io::Result<usize> {
    let chunk_size = chunks
        .iter()
        .fold(0, |ac, chunk: &PackageChunk| ac + chunk.raw_bytes.len());
    log::info!(
        "writing chunk to disk: {} files, {chunk_size} bytes",
        chunks.len()
    );
    package.write_chunks(output_file, chunks)
}

fn pack(
    config: &Config,
    input_dir: &path::Path,
    mut output_path: path::PathBuf,
) -> io::Result<(Duration, usize)> {
    if output_path.extension().is_none() {
        output_path = output_path.with_extension(EXTENSION);
    }
    let mut output_file = fs::File::create(output_path)?;
    let input_dir_name = input_dir.file_stem().expect("dir must have a valid stem");
    let mut bytes_written = 0;
    let time_t = Instant::now();
    // For now we'll just support 1 level of depth
    let mut package = PackageBuilder::new();
    bytes_written += package.write_header(&mut output_file)?;

    let files_to_process = input_dir.read_dir()?.filter_map(|file| {
        let file_path = file.expect("failed to read file in directory").path();
        if file_path.is_dir() {
            return None;
        }
        Some(file_path)
    }).collect::<Vec<_>>();

    let total_to_process = files_to_process.len();

    let mut chunks = vec![];
    for (i, file_path) in files_to_process.iter().enumerate() {
        if chunks.len() >= config.chunk_count {
            let flush = std::mem::take(&mut chunks);
            bytes_written += flush_chunks(&mut package, &mut output_file, flush)?;
        }

        // TODO: this sucks (utf-8 unwraps, push_str)
        let file_id = match file_path.file_name() {
            Some(stem) => {
                let mut str = String::new();
                str.push_str(input_dir_name.to_str().unwrap());
                str.push('/');
                str.push_str(stem.to_str().unwrap());
                str
            }
            None => {
                continue;
            }
        };
        log::info!("reading: {file_id} ({i}/{total_to_process})");
        let chunk = match fs::read(&file_path) {
            Ok(data) => PackageChunk::new_uncompressed(file_id, data),
            Err(e) => {
                log::warn!("failed to read: {file_path:?}, skipping: {e}");
                continue;
            }
        };
        chunks.push(chunk);
    }
    bytes_written += flush_chunks(&mut package, &mut output_file, chunks)?;
    bytes_written += package.write_toc(&mut output_file)?;

    Ok((time_t.elapsed(), bytes_written))
}


fn main() -> ExitCode {
    let matches = Command::new("packer")
        .about("Packs or unpacks files")
        .subcommand_required(true) // Must pick a subcommand
        .arg_required_else_help(true) // Show help if missing args
        .arg(Arg::new("quiet").long("quiet").short('q').help("Disables stdout").action(clap::ArgAction::SetTrue))
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
        quiet: false
    };

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
                Ok((time, written)) => {
                    log::info!("done, wrote {written} bytes in {}ms", time.as_millis())
                }
                Err(e) => {
                    log::error!("failed to write dir package: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        Some(("unpack", sub_m)) => {
            let input: &std::path::PathBuf = sub_m.get_one("input").unwrap();
            let output: &std::path::PathBuf = sub_m.get_one("output").unwrap();
            match unpack(input, output) {
                Ok((time, written)) => {
                    log::info!("done, wrote {written} bytes in {}ms", time.as_millis())
                }
                Err(e) => {
                    log::error!("failed to write dir package: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        _ => unreachable!("Subcommand required by clap"),
    }

    ExitCode::SUCCESS
}
