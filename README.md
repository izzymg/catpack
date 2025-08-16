# catpack


![catpack sprite](docs/ico.png)
Simple asset packer written in safe Rust inspired by [this video](https://www.youtube.com/watch?v=bMMOesLMWXs). 

### Features

* Supports [lz4](https://docs.rs/lz4_flex/latest/lz4_flex/index.html) compression
* Provides for streaming arbitrary data from the package
* Simple binary for packing & unpacking provided
* Zero dependencies without compression


### Library

Provides structures for writing and reading in package files. Useful for storing long-lived asset handles that seek to specific items in the package when needed, rather than
holding the entire package in memory.

### Binary

A simple pack/unpack utility, packs folders into a package file, or unpacks a package file back into a directory.

Also serves as a simple **example** for using the library.