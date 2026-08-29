# cuPQC SDK Applications

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20x86__64%20%7C%20ARM64-lightgrey)](https://developer.nvidia.com/cupqc)

This repository contains examples demonstrating the capabilities of NVIDIA cuPQC SDK.

**What's included:**
- Example applications with build configs
- Reference implementations


## About cuPQC SDK
NVIDIA cuPQC is a GPU-accelerated cryptography SDK containing specialized libraries for building high-performance cryptographic applications:

- **cuPQC-HASH** - Hash functions & Merkle Trees
- **cuPQC-PK** - Public-Key Cryptography (ML-KEM & ML-DSA)
- **cuPQC-BigInt** - Multi-Precision Integer Arithmetic
- **cuPQC-NTT** - Number Theoretic Transforms


## Applications & Examples

**Applications:** Browse [applications/](applications/) directory for application implementations.

**Examples:** Browse [examples/](examples/) directory for examples demonstrating individual SDK primitives:
- **Hash Functions** (SHA-2, SHA-3, Poseidon2, Merkle Trees)
- **Multi-Precision Integers** (wide addition, modular multiplication, pihex)
- **Number Theoretic Transforms** (standard and staged round-trips)
- **Public-Key Cryptography** (ML-KEM, ML-DSA)


## Quick Start
### Prerequisites

| Requirement | Specification |
|:------------|:--------------|
| **GPU** | Compute Capability 8.0+ (SM 80, 86, 87, 89, 90) |
| **CUDA** | 12.8 or newer |
| **Compiler** | C++17 (GCC 7+, Clang 9+) |
| **CMake** | 3.20+ (optional) |
| **SDK** | cuPQC 0.6.0+ |

### Download cuPQC SDK
[Download the SDK](https://developer.nvidia.com/cupqc-download/)

### Install cuPQC SDK

**x86_64:**
```bash
wget https://developer.download.nvidia.com/compute/cupqc/redist/cupqc/cupqc-sdk-0.6.0-x86_64.tar.gz
tar -xzf cupqc-sdk-0.6.0-x86_64.tar.gz
```

**ARM aarch64:**
```bash
wget https://developer.download.nvidia.com/compute/cupqc/redist/cupqc/cupqc-sdk-0.6.0-aarch64.tar.gz
tar -xzf cupqc-sdk-0.6.0-aarch64.tar.gz
```

### Configure SDK Path

Set the `CUPQC_SDK_DIR` environment variable to point to the extracted SDK directory:

```bash
# Option 1: Use extracted directory directly
export CUPQC_SDK_DIR=/path/to/cupqc-sdk-0.6.0-x86_64

# Option 2: Install to standard location
sudo mv cupqc-sdk-0.6.0-x86_64 /usr/local/cupqc-sdk
export CUPQC_SDK_DIR=/usr/local/cupqc-sdk
```

**Default path:** If `CUPQC_SDK_DIR` is not set, applications will look for the SDK at `/usr/local/cupqc-sdk`

**Make path persistent:** Add the `export` command to your `~/.bashrc` or `~/.zshrc`

**Verify installation:**
```bash
ls $CUPQC_SDK_DIR/include  # Should show header files
ls $CUPQC_SDK_DIR/lib      # Should show library files
```

For detailed installation, see the [Getting Started Guide](https://docs.nvidia.com/cuda/cupqc/guides/getting_started.html).


## Documentation & Resources
**Official Resources**
- [Product Page](https://developer.nvidia.com/cupqc) - Overview, features, and performance
- [Documentation](https://docs.nvidia.com/cuda/cupqc/) - Complete guides and API reference
- [Download Page](https://developer.nvidia.com/cupqc-download/) - Get the latest release
- [Developer Hub](https://nvidia.github.io/cuPQC/) - Blogs, examples, and application guides

**Community**
- [Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/quantum-computing/cupqc/703) - Community support
- [GitHub Issues](https://github.com/NVIDIA/cuPQC/issues) - Report bugs and request features

**Citation**
If you use cuPQC SDK in a publication, please cite it. Click the "Cite this repository" button in the About section above, or see [CITATION.cff](CITATION.cff) for details.


## License
This project is licensed under Apache 2.0. See [LICENSE](LICENSE) for details.

**Note:** These samples require the NVIDIA cuPQC SDK. By downloading and using the cuPQC SDK, you agree to fully comply with the terms and conditions of the [NVIDIA Software License Agreement](https://docs.nvidia.com/cuda/cupqc/additional/license.html).

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.
