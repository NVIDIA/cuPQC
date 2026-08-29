# Multi-Precision Integer Examples

This directory contains examples demonstrating the cuPQC-BigInt library, which provides GPU-optimized, fixed-width, unsigned multi-precision integer arithmetic. Unlike the other cuPQC libraries, cuPQC-BigInt is exposed entirely as `__device__` functions, so every operation is called from inside your own kernel rather than launched from the host. This makes it a building block for RSA and other classical public-key primitives, modular exponentiation, and any workload that needs integers wider than the native 32- and 64-bit hardware types.

**Requires cuPQC SDK 0.6.0 or newer**, the release that introduced cuPQC-BigInt.

## Examples

- **example_bigint_add.cu** - 1024-bit vector addition, with 4 warp lanes cooperating per value
- **example_bigint_addmulmod.cu** - 256-bit addition and modular multiplication in a single thread
- **example_bigint_pihex.cu** - 256 fractional hex digits of pi via Bellard's formula, using modular exponentiation, division, shifts, and a warp-level big-integer reduction

## Key Concepts

### Descriptors

A big-integer type is composed from operators, in the same style as the other cuPQC libraries:

```cpp
using BI256  = decltype(BitWidth<256>()  + SM<800>() + Thread());
using BI1024 = decltype(BitWidth<1024>() + SM<800>() + TPI<4>() + Warp());

using bigint      = typename BI256::bigint;       // 256-bit value
using bigint_wide = typename BI256::bigint_wide;  // 512-bit double-width value
```

- `BitWidth<BW>()` sets the width. `BW` must be a multiple of 32; the limb count is `BW / 32`.
- `Thread()` gives a single thread ownership of the whole value. `TPI` must be unset.
- `Warp()` spreads one value across `TPI<T>` consecutive warp lanes, which keeps register pressure and carry-propagation cost down for very wide integers. Exactly one of `Thread()` or `Warp()` is required.
- `SM<...>()` selects the target architecture and must match the architecture you compile for.

### Limb Layout

Values are arrays of little-endian 32-bit limbs, so limb 0 is least significant. Constructors and `store` optionally take an instance index, which makes a packed array of independent big integers loadable and storable directly by batch index:

```cpp
const BI256::bigint a(global_buf_a, index);  // load instance `index`
const auto sum = a + b;
sum.store(sums, index);                      // store to instance `index`
```

### Operand Widths

Both operands of a binary operation must be the same type. Nothing is implicitly promoted or truncated, so mixing widths is a compile error rather than a silent conversion. `div_rem` is the exception and accepts a narrower divisor. To widen a value explicitly, zero a buffer of the wider limb count, `store` the narrow value into its low limbs, and construct the wider type from that buffer.

## Building

```bash
make
```

This will build all examples in the directory. The Makefile expects cuPQC SDK at `/usr/local/cupqc-sdk` or a user-specified path (set `CUPQC_SDK_DIR` environment variable or modify the Makefile).

Note that cuPQC-BigInt is a static library built for a specific set of widths per `TPI`, and it is linked with LTO (`-dlto -lcupqc-bigint`). A configuration the type system accepts but the shipped library does not provide — an unsupported width, or a `Thread()`-only operation used on a warp descriptor — compiles cleanly and then fails to link.

### Matching the target architecture

The Makefile uses `-arch=native`, matching the other example directories. All three examples derive their `SM<...>()` operator from `__CUDA_ARCH__` so the descriptor stays in step with whatever architecture is actually being compiled:

```cpp
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define BIGINT_EXAMPLE_SM() + SM<900>()
// ... 890, 870, 860, 800 ...
#else
#define BIGINT_EXAMPLE_SM()
#endif

using BI256 = decltype(BitWidth<256>() + Thread() BIGINT_EXAMPLE_SM());
```

`SM<>` is optional in a bigint descriptor and defaults to `SM<800>`, which is what the host compilation pass and any unlisted architecture use. Only `BitWidth` is required to form a complete descriptor. The macro stops at 9.0 because that is the highest `SM<>` specialization the SDK provides, so a newer architecture falls through to the default rather than naming an `SM<>` that does not exist. cuPQC 0.6 itself targets compute capabilities 8.0, 8.6, 8.7, 8.9, and 9.0.

## Running

```bash
./example_bigint_add
./example_bigint_addmulmod
./example_bigint_pihex [start_bit]
```

`example_bigint_pihex` takes an optional nonnegative starting bit offset, defaulting to 0. At offset 0 it validates its output against the known leading hex digits of pi and reports GPU timings for the calculation, reduction, and copy-back phases.

## Further Reading

- [cuPQC-BigInt documentation](https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_bigint/cupqc_bigint_feature.html) - supported widths, the full operation list, and guidance on choosing a modular reduction path
