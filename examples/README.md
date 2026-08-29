# cuPQC SDK Examples

This directory contains examples demonstrating individual cuPQC SDK library functions. These examples showcase GPU-accelerated cryptographic primitives including hash functions, Merkle trees, multi-precision integer arithmetic, number theoretic transforms, and post-quantum public-key cryptography algorithms.

## Examples by Library

### Hash Functions (`hash/`)
The hash examples demonstrate various cryptographic hash functions and Merkle tree operations:
- **SHA-2** - SHA-256, a widely-used cryptographic hash function
- **SHA-3** - The latest NIST-standardized hash function
- **Poseidon2** - A zero-knowledge-friendly hash function optimized for ZK circuits
- **Merkle Trees** - Tree construction, proof generation, and verification with performance comparisons

### Multi-Precision Integers (`bigint/`)
The bigint examples demonstrate fixed-width, unsigned multi-precision integer arithmetic called directly from inside your own kernels:
- **Vector Addition** - 1024-bit addition with 4 warp lanes cooperating per value
- **Modular Multiplication** - 256-bit addition and `(a * b) mod m` in a single thread
- **pihex** - 256 fractional hex digits of pi via Bellard's formula, combining modular exponentiation, division, shifts, and a warp-level big-integer reduction

### Number Theoretic Transform (`ntt/`)
The NTT examples demonstrate GPU-accelerated cyclic transforms, which turn polynomial multiplication into cheap pointwise multiplication:
- **Standard NTT** - Batched forward and inverse round-trip, one polynomial per CUDA block
- **Staged NTT** - Large transforms split into two kernel passes with `SubSize<M>`, bounding each pass's shared-memory working set

### Public-Key Cryptography (`public_key/`)
The public-key examples demonstrate post-quantum cryptographic algorithms:
- **ML-KEM** - Module-Lattice Key Encapsulation Mechanism (NIST FIPS 203) for secure key exchange
- **ML-DSA** - Module-Lattice Digital Signature Algorithm (NIST FIPS 204) for digital signatures

## Building Examples

Each subdirectory includes a `Makefile`. To build all examples in a directory:

```bash
cd examples/<library>
make
```

The Makefiles expect cuPQC SDK at `/usr/local/cupqc-sdk` or a user-specified path (set `CUPQC_SDK_DIR` environment variable or modify the Makefile).
