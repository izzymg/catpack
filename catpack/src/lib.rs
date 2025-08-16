/**
    Write all the data into a single buffer.
    Store an offset to each file, along with some unique ID/string, etc.

*/
use std::{
    collections::HashMap,
    error::Error,
    fmt::{Debug, Display},
    fs,
    io::{self, BufReader, Read, Seek, Write},
    mem,
};

/// The size in bytes of the package format's header.
pub const HEADER_SIZE: usize = 4 + 4 + 4 + 4 + 8;

/// The size of the TOC-metadata
pub const TOC_START_SIZE: usize = 12;

const TOC_OFFSET_LOCATION: usize = 16;
const FILE_VERSION: u32 = 1;
const MAGIC: &[u8; 4] = b"CPKG";
const TOC_MAGIC: &[u8; 4] = b"toc!";
const TOC_ENTRY_MIN_SIZE: usize = 36;

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
    BadCompressionType(u32),
    UTF8(u64),
    EOF,
    IO(io::Error),
}

impl Display for PackageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic(actual) => {
                write!(
                    f,
                    "bad magic: {:?}",
                    std::str::from_utf8(actual).unwrap_or("invalid UTF-8")
                )
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
            Self::BadCompressionType(num) => {
                write!(f, "Bad compression type {num}")
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
    pub data_size: usize,
    /// The amount of data when uncompressed
    pub uncompressed_size: usize,
    /// What kind of (de)compression to apply when reading the data
    pub compression_type: CompressionType,
}

/// A package's table of contents (TOC). Describes what's in the file and how to get there, without holding the data directly.
#[derive(Debug)]
pub struct PackageToc {
    /// How many TOC entries are there?
    pub num_entries: u64,
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
            // u32: compression type
            reader.read_exact(&mut buffer[offset..offset + 4])?;
            let compression_type = parse_u32(&buffer[offset..offset + 4]);
            offset += 4;
            // u64: uncompressed size
            reader.read_exact(&mut buffer[offset..offset + 8])?;
            let uncompressed_size = parse_u64(&buffer[offset..offset + 8]);
            offset += 8;
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
                    data_size: data_size as usize,
                    file_offset,
                    compression_type: compression_type.try_into()?,
                    uncompressed_size: uncompressed_size as usize,
                },
            );

            file_index += 1;
        }

        Ok(Self {
            num_entries,
            entries,
        })
    }

    /// Wraps `data` in a `BufReader` and calls `Self::from_reader`
    pub fn from_memory(data: &[u8], buffer: &mut Vec<u8>) -> Result<Self, PackageError> {
        let mut reader = BufReader::new(data);
        Self::from_reader(&mut reader, buffer)
    }

    /// Returns the TOC entry at the given file identifier.
    pub fn get_entry(&self, name: &str) -> Option<&PackageTocEntry> {
        self.entries.get(name)
    }
}

/// Allows for reading
pub struct PackageSeeker {
    pub handle: BufReader<fs::File>,
}

impl PackageSeeker {
    /// Reads the data using given TOC entry into `buffer`. Does not apply decompression to the buffer.
    pub fn read_data_raw(
        &mut self,
        entry: &PackageTocEntry,
        buffer: &mut Vec<u8>,
    ) -> io::Result<()> {
        if buffer.len() < entry.data_size {
            buffer.resize(entry.data_size, 0u8);
        }
        self.handle.seek(io::SeekFrom::Start(entry.file_offset))?;
        self.handle.read_exact(&mut buffer[0..entry.data_size])?;


        Ok(())
    }

    /// Reads the data using given TOC entry into `buffer`. Applies decompression to the buffer.
    /// Resizes the buffer if it's too small for the decompressed data.
    pub fn read_data(&mut self, entry: &PackageTocEntry, buffer: &mut Vec<u8>) -> io::Result<()> {
        self.read_data_raw(entry, buffer)?;
        if let Some(mut result) =
            decompress_data(buffer, &entry.compression_type, entry.uncompressed_size)
        {
            std::mem::swap(buffer, &mut result);
        }
        Ok(())
    }
}

pub struct PackageHandle {
    pub header: PackageHeader,
    pub toc: PackageToc,
    pub seeker: PackageSeeker,
}

impl PackageHandle {
    /// Consumes the given catpack file to create a `PackageHandle`
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

    /// Consumes a catpack package file to create a `PackageHandle`
    /// This validates the package and creates a handle for reading its contents, without loading the data segment into memory.
    pub fn from_file(file: fs::File) -> Result<Self, PackageError> {
        // Reasonable default buffer size
        Self::from_file_with_buffer(
            file,
            &mut Vec::with_capacity(HEADER_SIZE + TOC_START_SIZE + 1000),
        )
    }

    pub fn iter(&mut self) -> impl Iterator<Item = &String> {
        self.toc.entries.keys()
    }

    /// Reads a given file by its identifier into `buffer`, which will be resized if needed.
    /// If `None` is returned, there was no entry for the given identifier in the package's table of contents.
    pub fn read_by_id(&mut self, name: &str, buffer: &mut Vec<u8>) -> Option<io::Result<()>> {
        let entry = self.toc.get_entry(name)?;
        match self.seeker.read_data(entry, buffer) {
            Ok(()) => Some(Ok(())),
            Err(e) => Some(Err(e)),
        }
    }

    /// Returns the size of the data at the given identifier without reading the data.
    /// This is the size of the data in the package, i.e. with compression applied if applicable.
    /// To allocate a buffer for reading, you likely want `get_read_size`.
    /// Returns `None` if there is no table of contents entry for the given identifier.
    pub fn get_compressed_data_size(&self, name: &str) -> Option<usize> {
        let entry = self.toc.get_entry(name)?;
        Some(entry.data_size)
    }

    /// Returns the known uncompressed/raw size of the data at the given identifier, without reading the data.
    /// Returns `None` if there is no table of contents entry for the given identifier.
    pub fn get_read_size(&self, name: &str) -> Option<usize> {
        let entry = self.toc.get_entry(name)?;
        Some(entry.uncompressed_size)
    }
}

/* Writing */

/// A chunk of data to write into the package.
#[derive(Debug)]
pub struct PackageChunk {
    pub raw_bytes: Vec<u8>,
    /// This chunk's bytes will be passed through the compression algorithm before write.
    pub compression_type: CompressionType,
    pub id: String,
}

impl PackageChunk {
    pub fn new(id: String, data: Vec<u8>, apply_compression: CompressionType) -> Self {
        Self {
            id,
            raw_bytes: data,
            compression_type: apply_compression,
        }
    }

    pub fn new_uncompressed(id: String, data: Vec<u8>) -> Self {
        Self {
            id,
            raw_bytes: data,
            compression_type: CompressionType::None,
        }
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.raw_bytes
    }
}

#[must_use]
fn compress_data(data: &Vec<u8>, compression_type: &CompressionType) -> Option<Vec<u8>> {
    match compression_type {
        CompressionType::None => None,
        #[cfg(feature = "lz4")]
        CompressionType::LZ4 => {
            use lz4_flex;
            let compressed_data = lz4_flex::block::compress(&data);
            Some(compressed_data)
        }
    }
}

#[must_use]
fn decompress_data(
    data: &Vec<u8>,
    compression_type: &CompressionType,
    uncompressed_size: usize,
) -> Option<Vec<u8>> {
    match compression_type {
        CompressionType::None => None,
        #[cfg(feature = "lz4")]
        CompressionType::LZ4 => {
            use lz4_flex;
            Some(
                lz4_flex::block::decompress(&data, uncompressed_size )
                .expect("decompression failed, package is likely corrupt due to invalid uncompressed size header")
            )
        }
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub enum CompressionType {
    #[default]
    None,

    #[cfg(feature = "lz4")]
    LZ4 = 1,
}

impl CompressionType {
    fn to_le_bytes(&self) -> [u8; 4] {
        match self {
            CompressionType::None => 0u32.to_le_bytes(),
            #[cfg(feature = "lz4")]
            CompressionType::LZ4 => 1u32.to_le_bytes(),
        }
    }
}

impl TryFrom<u32> for CompressionType {
    type Error = PackageError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            #[cfg(feature = "lz4")]
            1 => Ok(Self::LZ4),
            _ => Err(PackageError::BadCompressionType(value)),
        }
    }
}

#[derive(Debug)]
struct PackageBuilderEntry {
    id: String,
    data_size: usize,
    uncompressed_size: usize,
    file_offset: usize,
    compression: CompressionType,
}

#[derive(Default)]
pub struct WriteResult {
    pub total_compressed_size: usize,
    pub total_uncompressed_size: usize,
}

impl std::ops::Add for WriteResult {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            total_compressed_size: self.total_compressed_size + rhs.total_compressed_size,
            total_uncompressed_size: self.total_uncompressed_size + rhs.total_uncompressed_size,
        }
    }
}

/// Allows for building and writing out a package.
/// Currently requires the user to hold their entire package in-memory - TODO: buffered/flush writing.
#[derive(Debug, Default)]
pub struct PackageBuilder {
    toc: Vec<PackageBuilderEntry>,
    // How much of the data section has been written so far
    data_size: usize,
}

impl PackageBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes the header data with an empty TOC offset to the writer.
    pub fn write_header<W>(&self, writer: &mut W) -> io::Result<usize>
    where
        W: Write,
    {
        let mut written = 0;
        let mut header = vec![];
        // Magic
        header.extend(MAGIC);
        // File version (u32)
        header.extend(&FILE_VERSION.to_le_bytes());
        // Flags (u32)
        header.extend(&0u32.to_le_bytes());
        // Reserved/padding
        header.extend(&0u32.to_le_bytes());
        // Offset to TOC (u64) - for now we write 0 as we don't know how much data there is.
        header.extend(&0u64.to_le_bytes());
        written += writer.write(&header)?;
        assert_eq!(written, HEADER_SIZE);
        Ok(written)
    }

    /// Writes the given data entries to the file, keeping track of data for the TOC.
    pub fn write_chunks<W>(
        &mut self,
        writer: &mut W,
        datas: Vec<PackageChunk>,
    ) -> io::Result<WriteResult>
    where
        W: Write + Seek,
    {
        let mut total_written: usize = 0;
        let mut total_uncompressed: usize = 0;

        let mut entries = vec![];
        writer.seek(io::SeekFrom::End(0))?;

        for data in datas.into_iter() {
            // Calculate where this piece of data starts in the file
            let file_offset = HEADER_SIZE + self.data_size;

            // Snag the uncompressed size, then try compress it
            let compression_type = data.compression_type;
            let id = data.id.clone();
            let uncompressed_size = data.raw_bytes.len();
            total_uncompressed += uncompressed_size;
            let raw_bytes = data.into_inner();

            let write_data = compress_data(&raw_bytes, &compression_type).unwrap_or(raw_bytes);
            self.data_size += write_data.len();

            // Store the data for writing the TOC at the end
            entries.push(PackageBuilderEntry {
                id,
                file_offset,
                data_size: write_data.len(),
                uncompressed_size,
                compression: compression_type,
            });

            // Write file data to the end of the file
            total_written += writer.write(&write_data)?;
        }
        self.toc.extend(entries);
        // Write the new position of the offset
        writer.seek(io::SeekFrom::Start(TOC_OFFSET_LOCATION as u64))?;
        let toc_offset = HEADER_SIZE + self.data_size;
        writer.write(&toc_offset.to_le_bytes())?;
        Ok(WriteResult {
            total_compressed_size: total_written,
            total_uncompressed_size: total_uncompressed,
        })
    }

    /// Writes the stored TOC to the end of the file.
    pub fn write_toc<W>(&self, writer: &mut W) -> io::Result<usize>
    where
        W: Write + Seek,
    {
        writer.seek(io::SeekFrom::End(0))?;
        // TOC magic + size header
        let mut table_of_contents = Vec::new();
        table_of_contents.extend(TOC_MAGIC);
        table_of_contents.extend(&(self.toc.len() as u64).to_le_bytes());

        for entry in self.toc.iter() {
            // u32 file compression type
            table_of_contents.extend(entry.compression.to_le_bytes());
            // u64 file uncompressed size
            table_of_contents.extend(entry.uncompressed_size.to_le_bytes());
            // u64 filename length
            let filename_len = entry.id.len();
            table_of_contents.extend(filename_len.to_le_bytes());
            // filename
            table_of_contents.extend(entry.id.as_bytes());
            // u64 data size
            table_of_contents.extend(entry.data_size.to_le_bytes());
            // u64 offset
            table_of_contents.extend(entry.file_offset.to_le_bytes());
        }

        writer.write(&table_of_contents)
    }
}

/* Testing */

#[cfg(test)]
mod test {
    use super::*;

    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    #[test]
    fn test_write_package() {
        let mut output_file = fs::File::create("test/out.cpkg").unwrap();
        let total_t = Instant::now();
        let mut package = PackageBuilder::new();
        package
            .write_header(&mut output_file)
            .expect("header must write");

        let item1 = "hello world";
        let item2 = (0..10)
            .flat_map(|v| (v as u32 * v as u32).to_le_bytes())
            .collect::<Vec<_>>();

        let item3 = "woah";

        let chunks = vec![
            PackageChunk::new_uncompressed("item1".to_string(), item1.as_bytes().to_vec()),
            PackageChunk::new_uncompressed("where/item/be/item2".to_string(), item2.to_vec()),
            PackageChunk::new_uncompressed(
                "look_at_this/photo.graph".to_string(),
                item3.as_bytes().to_vec(),
            ),
        ];

        let chunk_t = Instant::now();
        let written = package
            .write_chunks(&mut output_file, chunks)
            .expect("chunks must write");

        println!(
            "wrote chunks ({} bytes) ({} uncompressed) in: {:?}",
            written.total_compressed_size,
            written.total_uncompressed_size,
            chunk_t.elapsed()
        );

        package
            .write_toc(&mut output_file)
            .expect("file should write");
        println!("wrote package in: {:?}", total_t.elapsed());
    }

    #[test]
    fn test_check_header() {
        let data = fs::read(Path::new("test/in.cpkg")).expect("file must load");
        PackageHeader::read_from_memory(&data).expect("must get header data");
    }

    #[test]
    fn test_load_package() {
        let file = fs::File::open(Path::new("test/in.cpkg")).expect("file must open");
        let mut pkg = PackageHandle::from_file(file).expect("package must parse");
        println!("header: {:?}", pkg.header);
        println!("toc: {:?}", pkg.toc);

        let file_size = pkg
            .get_compressed_data_size("item1")
            .expect("item1 must have size");
        assert_eq!(file_size, "hello world".len());
        let mut buffer = vec![0_u8; file_size];
        pkg.read_by_id("item1", &mut buffer)
            .expect("file must exist")
            .expect("file read must succeed");
        let str = String::from_utf8(buffer).expect("must be valid utf8");
        assert_eq!(str, "hello world");
    }
}
