# catpack

Simple asset packer written in safe Rust inspired by: https://www.youtube.com/watch?v=bMMOesLMWXs

### Library

Provides structures for writing and reading in package files. Useful for storing long-lived asset handles that seek to specific items in the package when needed, rather than
holding the entire package in memory.

### Binary

A simple pack/unpack utility, packs folders into a package file, or unpacks a package file back into a directory.

Also serves as a simple example for using the library.