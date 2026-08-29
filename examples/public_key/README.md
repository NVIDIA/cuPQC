# Public-Key Cryptography Examples

This directory contains examples demonstrating the cuPQC-PK library, which provides GPU-accelerated implementations of post-quantum cryptographic algorithms.

## Examples

- **example_ml_kem.cu** - ML-KEM (Key Encapsulation Mechanism)
- **example_ml_dsa.cu** - ML-DSA (Digital Signature Algorithm)

## Building

```bash
make
```

This will build all examples in the directory. The Makefile expects cuPQC SDK at `/usr/local/cupqc-sdk` or a user-specified path (set `CUPQC_SDK_DIR` environment variable or modify the Makefile).

## Running

```bash
./example_<name>
```
