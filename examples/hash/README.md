# Hash Function Examples

This directory contains examples demonstrating the cuPQC-HASH library, which provides GPU-accelerated implementations of cryptographic hash functions and Merkle trees. Hash functions are fundamental building blocks in cryptography, used for data integrity verification, digital signatures, and zero-knowledge proof systems.

## Examples

- **example_sha2.cu** - SHA-2 (SHA-256) hashing
- **example_sha3.cu** - SHA-3 hashing
- **example_poseidon2.cu** - Poseidon2 ZK-friendly hash function
- **example_merkle.cu** - Merkle tree construction, proof generation, and verification
- **example_merkle_multi_block.cu** - Large-scale Merkle tree generation using parallel processing

## Building

```bash
make
```

This will build all examples in the directory. The Makefile expects cuPQC SDK at `/usr/local/cupqc-sdk` or a user-specified path (set `CUPQC_SDK_DIR` environment variable or modify the Makefile).

## Running

```bash
./example_<name>
```

Each example demonstrates different aspects of hash functions and Merkle trees, with detailed output showing the operations performed and results.
