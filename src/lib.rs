/**
    Write all the data into a single buffer.
    Store an offset to each file, along with some unique ID/string, etc.

    Assumes LE architecture.
*/

#[cfg(target_endian = "big")]
compile_error!("BE not supported!");

use std::{
    collections::HashMap,
    error::Error,
    fmt::{Debug, Display},
    fs,
    io::{self, BufRead, BufReader, Read, Seek, Write},
    mem, path,
};

/// The size in bytes of the package format's header.
pub const HEADER_SIZE: usize = 4 + 4 + 4 + 4 + 8;

/// The size of the TOC-metadata
pub const TOC_START_SIZE: usize = 20;

const FILE_VERSION: u32 = 1;
const MAGIC: &[u8; 4] = b"DPKG";
const TOC_MAGIC: &[u8; 4] = b"toc!";
const TOC_ENTRY_MIN_SIZE: usize = 24;

fn parse_u32(input: &[u8]) -> u32 {
    u32::from_le_bytes(input.try_into().expect("slice should be exactly 4 bytes"))
}

fn parse_u64(input: &[u8]) -> u64 {
    u64::from_le_bytes(input.try_into().expect("slice should be exactly 8 bytes"))
}

fn check_magic(input: &[u8], expected: &[u8; 4]) -> bool {
    input.len() >= 4 && &input[0..4] == expected
}

#[derive(Debug)]
pub enum PackageError {
    BadMagic(&'static [u8]),
    WrongSize(usize),
    WrongVersion(u32),
    UTF8(u64),
    EOF,
    IO(io::Error),
}

impl Display for PackageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic(actual) => {
                write!(f, "bad magic: {:?}", std::str::from_utf8(actual).unwrap_or("invalid UTF-8"))
            }
            Self::WrongSize(actual) => {
                write!(f, "invalid header size {actual} expected {HEADER_SIZE}")
            }
            Self::WrongVersion(actual) => {
                write!(f, "invalid version {actual} for max {FILE_VERSION}")
            }
            Self::UTF8(toc_offset) => {
                write!(f, "UTF-8 parse error at TOC entry: {toc_offset}")
            }
            Self::EOF => {
                write!(f, "Encountered unexpected EOF")
            }
            Self::IO(err) => write!(f, "io: {err}"),
        }
    }
}

impl From<io::Error> for PackageError {
    fn from(value: io::Error) -> Self {
        Self::IO(value)
    }
}

/* Reading */

impl Error for PackageError {}

/// The header data from a package file.
#[derive(Debug, Copy, Clone)]
pub struct PackageHeader {
    pub version: u32,
    pub flags: u32,
    pub reserved: u32,
    pub toc_offset: u64,
}

impl PackageHeader {
    /// Attempts to read a package header from the given reader. Throws errors if there are mismatches in the expected layout.
    /// Uses the given buffer to read data into.
    pub fn read(reader: &mut impl Read, buffer: &mut Vec<u8>) -> Result<Self, PackageError> {
        buffer.clear();
        buffer.resize(HEADER_SIZE, 0);
        reader.read_exact(buffer)?;
        if !check_magic(buffer, MAGIC) {
            return Err(PackageError::BadMagic(MAGIC));
        }
        let version = parse_u32(&buffer[4..8]);
        let flags = parse_u32(&buffer[8..12]);
        let reserved = parse_u32(&buffer[12..16]);
        let toc_offset = parse_u64(&buffer[16..24]);

        if version > FILE_VERSION {
            return Err(PackageError::WrongVersion(version));
        }

        Ok(Self {
            version,
            flags,
            reserved,
            toc_offset,
        })
    }

    /// Attempts to read a package header from the given dataset. Throws errors if there are mismatches in the expected layout.
    /// Does not check the toc offset validity as that would require assuming `data` is the entire file.
    pub fn read_from_memory(data: &[u8]) -> Result<Self, PackageError> {
        if data.len() < HEADER_SIZE {
            return Err(PackageError::WrongSize(data.len()));
        }
        if !check_magic(data, MAGIC) {
            return Err(PackageError::BadMagic(MAGIC));
        }
        let version = parse_u32(&data[4..8]);
        let flags = parse_u32(&data[8..12]);
        let reserved = parse_u32(&data[12..16]);
        let toc_offset = parse_u64(&data[16..24]);

        if version > FILE_VERSION {
            return Err(PackageError::WrongVersion(version));
        }

        Ok(Self {
            version,
            flags,
            reserved,
            toc_offset,
        })
    }
}

/// A single entry in the package's TOC.
/// To read an entry out of a package, we read `package_data[file_offset..file_offset+data_size]`
#[derive(Debug)]
pub struct PackageTocEntry {
    /// Where in the package file to seek from when reading.
    pub file_offset: u64,
    /// The amount of data this entry's data section is.
    pub data_size: u64,
}

/// A package's table of contents (TOC). Describes what's in the file and how to get there, without holding the data directly.
#[derive(Debug)]
pub struct PackageToc {
    /// How many TOC entries are there?
    pub num_entries: u64,
    pub reserved: u64,
    /// Maps data identifiers to TOC entries
    pub entries: HashMap<String, PackageTocEntry>,
}

impl PackageToc {
    /// Reads the TOC from the given data.
    /// Expects the reader to start *from* the start of the TOC (i.e. offset by header->toc_offset);
    /// Typically you wouldn't call this directly, but prefer to use a `PackageHandle`.
    pub fn from_reader(reader: &mut impl Read, buffer: &mut Vec<u8>) -> Result<Self, PackageError> {
        buffer.clear();
        buffer.resize(TOC_START_SIZE, 0);
        reader.read_exact(buffer)?;
        // Parse TOC magic
        if !check_magic(buffer, TOC_MAGIC) {
            return Err(PackageError::BadMagic(TOC_MAGIC));
        }
        let num_entries = parse_u64(&buffer[4..12]);
        let reserved = parse_u64(&buffer[12..20]);

        let mut offset: usize = TOC_START_SIZE;
        let mut entries = HashMap::new();
        let mut file_index = 0;

        // Try to predict the needed size
        // TODO: allow user to influence this prediction?
        let predicted_filename_size = mem::size_of::<char>() * 30;
        let est_files = 100;
        let est_size = offset + (predicted_filename_size + TOC_ENTRY_MIN_SIZE) * est_files;
        if buffer.len() < est_size {
            buffer.resize(est_size, 0);
        }

        while file_index < num_entries {
            let required_size = offset + TOC_ENTRY_MIN_SIZE;
            if buffer.len() <= required_size {
                buffer.resize(required_size, 0);
            }
            reader.read_exact(&mut buffer[offset..offset + 8])?;
            // u64: filename size
            let filename_size = parse_u64(&buffer[offset..offset + 8]);
            let required_size = offset + TOC_ENTRY_MIN_SIZE + filename_size as usize;
            if buffer.len() <= required_size {
                buffer.resize(required_size, 0);
            }
            offset += 8;
            // Find the size of the rest of the entry, then read it
            let next_chunk_size = filename_size as usize + 8 + 8;
            reader.read_exact(&mut buffer[offset..offset + next_chunk_size])?;
            // filename
            let filename_bytes = &buffer[offset..offset + filename_size as usize];
            let filename = String::from_utf8(filename_bytes.to_vec())
                .map_err(|_| PackageError::UTF8(file_index))?;
            offset += filename_size as usize;
            // u64 data size
            let data_size = parse_u64(&buffer[offset..offset + 8]);
            offset += 8;
            // u64 offset
            let file_offset = parse_u64(&buffer[offset..offset + 8]);
            offset += 8;

            entries.insert(
                filename,
                PackageTocEntry {
                    data_size,
                    file_offset,
                },
            );

            file_index += 1;
        }

        Ok(Self {
            num_entries,
            reserved,
            entries,
        })
    }

    /// Wraps `data` in a `BufReader` and calls `Self::from_reader`
    pub fn from_memory(data: &[u8], buffer: &mut Vec<u8>) -> Result<Self, PackageError> {
        let mut reader = BufReader::new(data);
        Self::from_reader(&mut reader, buffer)
    }
}

/// Allows for reading
pub struct PackageSeeker {
    pub handle: BufReader<fs::File>,
}

// TODO: take TOC in and find actual data :o
impl PackageSeeker {}

pub struct PackageHandle {
    pub header: PackageHeader,
    pub toc: PackageToc,
    pub seeker: PackageSeeker,
}

impl PackageHandle {
    /// Consumes the file to create a `PackageHandle`
    /// Uses the given buffer to allocate data into: must be greater than `HEADER_SIZE` + `TOC_START_SIZE`
    pub fn from_file_with_buffer(
        file: fs::File,
        buffer: &mut Vec<u8>,
    ) -> Result<Self, PackageError> {
        let mut buf_reader = BufReader::new(file);
        let header = PackageHeader::read(&mut buf_reader, buffer)?;

        buf_reader.seek(io::SeekFrom::Start(header.toc_offset))?;

        let toc = PackageToc::from_reader(&mut buf_reader, buffer)?;

        let file = buf_reader.into_inner();
        let seeker = PackageSeeker {
            handle: BufReader::new(file),
        };

        Ok(Self {
            header,
            toc,
            seeker,
        })
    }

    /// Consumes the file to create a `PackageHandle`
    pub fn from_file(file: fs::File) -> Result<Self, PackageError> {
        // Reasonable default buffer size
        Self::from_file_with_buffer(
            file,
            &mut Vec::with_capacity(HEADER_SIZE + TOC_START_SIZE + 1000),
        )
    }

    pub fn header(&self) -> &PackageHeader {
        &self.header
    }
}

/// Holds data in the `PackageBuilder` for writing to disk.
struct PackageData {
    data: Vec<u8>,
    name: String,
}

impl Debug for PackageData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "package: name: {:?} data size: {:?}",
            self.name,
            self.data.len()
        )
    }
}

/* Writing */

/// Allows for building and writing out a package.
/// Currently requires the user to hold their entire package in-memory - TODO: buffered/flush writing.
#[derive(Debug, Default)]
pub struct PackageBuilder {
    entries: Vec<PackageData>,
}

impl PackageBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Consumes self, writing out the package to the filesystem at `output_path`
    pub fn write(self, output_path: &path::Path) -> io::Result<u64> {
        // Open the file for writing
        let mut file = fs::File::create(output_path)?;
        let mut total_written: u64 = 0;

        // Build TOC offset (header size + all file data size)
        let mut toc_offset: u64 = HEADER_SIZE as u64;
        let mut entry_offsets = vec![0_u64; self.entries.len()];
        for (i, entry) in self.entries.iter().enumerate() {
            let entry_data_size = entry.data.len() as u64;
            toc_offset += entry_data_size;
            entry_offsets[i] = entry_data_size;
        }

        // magic + flags + version 32 bit integers
        let mut header: Vec<u8> = Vec::with_capacity(mem::size_of::<u32>() * 3);
        // Magic
        header.extend(MAGIC);
        // File version (u32)
        header.extend(&FILE_VERSION.to_le_bytes());
        // Flags (u32)
        header.extend(&0u32.to_le_bytes());
        // Reserved/padding
        header.extend(&0u32.to_le_bytes());
        // Offset to TOC (u64)
        header.extend(&toc_offset.to_le_bytes());

        total_written += file.write(&header)? as u64;
        assert_eq!(total_written, HEADER_SIZE as u64);
        for entry in self.entries.iter() {
            let bytes = file.write(&entry.data)? as u64;
            total_written += bytes;
        }

        // TOC magic + size header
        let mut table_of_contents = Vec::new();
        table_of_contents.extend(TOC_MAGIC);
        table_of_contents.extend(&(self.entries.len() as u64).to_le_bytes());
        table_of_contents.extend(&0u64.to_le_bytes()); // Reserved

        for (i, entry) in self.entries.iter().enumerate() {
            // u64 filename length
            let filename_len = entry.name.len();
            table_of_contents.extend(filename_len.to_le_bytes());
            // filename
            table_of_contents.extend(entry.name.as_bytes());
            // u64 data size
            table_of_contents.extend(entry.data.len().to_le_bytes());
            // u64 offset
            table_of_contents.extend(entry_offsets[i].to_le_bytes());
        }

        total_written += file.write(&table_of_contents)? as u64;
        Ok(total_written)
    }

    pub fn add(&mut self, name: String, data: Vec<u8>) {
        let entry = PackageData { name, data };
        self.entries.push(entry);
    }
}

/* Testing */

#[cfg(test)]
mod test {
    use super::*;

    use path::Path;
    use std::time::Instant;

    #[test]
    fn test_write_package() {
        let mut package = PackageBuilder::new();

        let item1 = "hello world";
        let item2 = (0..10)
            .into_iter()
            .flat_map(|v| (v as u32 * v as u32).to_le_bytes())
            .collect::<Vec<_>>();

        let item3 = "woah";

        package.add("item1".to_string(), item1.as_bytes().to_vec());
        package.add("where/item/be/item2".to_string(), item2.to_vec());
        package.add(
            "/look_at_this/photo.graph".to_string(),
            item3.as_bytes().to_vec(),
        );

        let start = Instant::now();

        package
            .write(Path::new("test/out.dpk"))
            .expect("file should write");

        println!("wrote package in: {:?}", start.elapsed());
    }

    #[test]
    fn test_check_header() {
        let data = fs::read(Path::new("test/in.dpk")).expect("file must load");
        PackageHeader::read_from_memory(&data).expect("must get header data");
    }

    #[test]
    fn test_load_package() {
        let file = fs::File::open(Path::new("test/in.dpk")).expect("file must open");
        let pkg = PackageHandle::from_file(file).expect("package must parse");
        println!("header: {:?}", pkg.header);
        println!("toc: {:?}", pkg.toc);
    }
}
