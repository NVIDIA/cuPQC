# Merkle Tree Membership Proof (Poseidon2 BabyBear)

Complete membership proof system using Merkle trees with Poseidon2 (BabyBear field). When run, the application generates a Merkle tree (2M+ leaves), creates a membership proof, verifies it, and demonstrates tampering detection.

## Building

```bash
# Build
make

# Run
./merkle_proof
```

**Note:** The Makefile expects cuPQC SDK at `/usr/local/cupqc-sdk` or a user-specified path. If your SDK is installed elsewhere:
- Set the `CUPQC_SDK_DIR` environment variable, or
- Modify `CUPQC_SDK_DIR` in the Makefile

## cuPQC Primitives Used

| Primitive | Description |
|-----------|-------------|
| **Poseidon2 (BabyBear)** | ZK-friendly hash function used for leaf hashing |
| **Merkle Trees** | Tree generation, membership proof creation, and verification |

## Documentation

- [cuPQC Hash Documentation](https://docs.nvidia.com/cuda/cupqc/guides/cupqc_hash_usage.html)
- [Merkle Tree API Reference](https://docs.nvidia.com/cuda/cupqc/api/cupqc_hash_api.html#merkle-trees)
- [Poseidon2 API Reference](https://docs.nvidia.com/cuda/cupqc/api/cupqc_hash_api.html#poseidon2)

