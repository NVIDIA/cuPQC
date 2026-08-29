# Number Theoretic Transform Examples

This directory contains examples demonstrating the cuPQC-NTT library, which provides GPU-accelerated cyclic Number Theoretic Transforms. The NTT is the modular-arithmetic analogue of the FFT: it turns polynomial multiplication into cheap pointwise multiplication, which is why it sits on the critical path of lattice-based post-quantum schemes, fully homomorphic encryption, and zero-knowledge proof systems.

**Requires cuPQC SDK 0.6.0 or newer**, the release that introduced cuPQC-NTT.

## Examples

- **example_ntt.cu** - Batched forward and inverse NTT round-trip at N=1024, one polynomial per CUDA block
- **example_staged_ntt.cu** - Batched staged round-trip at N=16384, decomposed into two kernel passes with `SubSize<256>`

Both examples verify that `INTT(NTT(f)) == f` for every coefficient of every polynomial in the batch and exit non-zero if any coefficient does not match.

## Key Concepts

### Descriptors

An NTT type is composed from operators, in the same style as the other cuPQC libraries:

```cpp
using ForwardNTT = decltype(Algorithm<algorithm::NTT>()
                            + Direction<nttDirection::FORWARD>()
                            + Precision<uint32_t>()
                            + Size<1024>()
                            + Block()
                            + BlockDim<128>());
```

`Direction` selects forward or inverse, `Precision` selects the coefficient and modulus width (`uint16_t`, `uint32_t`, or `uint64_t`), `Size<N>` sets the transform length, and `Block()` with `BlockDim<...>` requests block execution with a given thread count.

### Montgomery Domain

Device transforms operate on Montgomery-domain data, so conversion happens at the memory boundary rather than per operation. A typical kernel loads into shared memory, executes, and stores back:

```cpp
ForwardNTT().load_to_mont(sdata, poly, ntt_const);
__syncthreads();
ForwardNTT().execute(sdata, twiddles, ntt_const.p);
__syncthreads();
ForwardNTT().store_from_mont(sdata, poly, ntt_const);
```

### Twiddle Tables

Twiddle factors are generated once on the device with `make_twiddles`, converted with `transform_twiddles_to_mont`, and then reused across the entire batch. The forward table uses a primitive N-th root of unity `g`; the inverse table uses `g^-1 mod p`. The inverse transform additionally takes `N^-1 mod p` to normalize its output.

### Staged Transforms

For large `N`, a single block's shared memory cannot hold the full workspace. `SubSize<M>` splits the transform into two kernel passes with `K = N / M`:

- Forward stage 1 runs `M` blocks, each a `K`-point transform over contiguous elements.
- Forward stage 2 runs `K` blocks, each an `M`-point transform over strided elements.

The inverse reverses the stage ordering and applies the `1/N` normalization in stage 2. Each pass bounds its per-block working set to `max(K, M)` elements rather than the full `N`. Staged transforms cover `2^14` through `2^24` points, with the allowed `SubSize<M>` values narrowing as `N` grows.

### Fields

`BabyBear` and `KoalaBear` are built in, along with precomputed roots named `*_primitive_root_S` for transform sizes `2^S` where `10 <= S <= 24`. Custom primes up to 62 bits are also supported. Both examples use KoalaBear, `p = 2^31 - 2^24 + 1`.

## Building

```bash
make
```

This will build all examples in the directory. The Makefile expects cuPQC SDK at `/usr/local/cupqc-sdk` or a user-specified path (set `CUPQC_SDK_DIR` environment variable or modify the Makefile).

cuPQC-NTT is a static library linked with LTO (`-dlto -lcupqc-ntt`), built for a specific set of transform sizes and sub-sizes. A configuration the type system accepts but the shipped library does not provide compiles cleanly and then fails to link.

## Running

```bash
./example_ntt
./example_staged_ntt
```

## Further Reading

- [cuPQC-NTT documentation](https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_ntt/cupqc_ntt_feature.html) - supported transform sizes, prime fields, and the staged decomposition rules
