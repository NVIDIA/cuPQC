/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * cuPQC BigInt pihex Example
 *
 * Computes 256 fractional hex digits of pi starting at an arbitrary bit
 * offset, using Bellard's formula evaluated entirely in big-integer
 * arithmetic on the GPU.
 *
 * Bellard's formula writes pi as seven alternating-sign subterms:
 *
 *   pi = 2^-6 sum(n >= 0) (-1)^n 2^(-10n) [
 *          -2^5/(4n+1) - 1/(4n+3)
 *          +2^8/(10n+1) - 2^6/(10n+3)
 *          -2^2/(10n+5) - 2^2/(10n+7) + 1/(10n+9) ].
 *
 * Carry 2^-6 and (-1)^n 2^(-10n) into each subterm, then split
 * n into its even and odd cases, n=2m and n=2m+1.  Put the positive
 * case first and the negative case second.  The seven original subterms
 * become these seven fixed-sign pairs (fourteen subterms total):
 *
 *   pi = sum(m >= 0) [
 *       ( 2^(-20m-16+5)/(8m+5)   - 2^(-20m-6+5)/(8m+1) )
 *     + ( 2^(-20m-16)/(8m+7)     - 2^(-20m-6)/(8m+3) )
 *     + ( 2^(-20m-6+8)/(20m+1)   - 2^(-20m-16+8)/(20m+11) )
 *     + ( 2^(-20m-16+6)/(20m+13) - 2^(-20m-6+6)/(20m+3) )
 *     + ( 2^(-20m-16+2)/(20m+15) - 2^(-20m-6+2)/(20m+5) )
 *     + ( 2^(-20m-16+2)/(20m+17) - 2^(-20m-6+2)/(20m+7) )
 *     + ( 2^(-20m-6)/(20m+9)     - 2^(-20m-16)/(20m+19) ) ].
 *
 * For one positive subterm 2^p/denom, shifting pi by start_bit and
 * retaining B=320 fractional bits asks for
 *
 *   floor(2^pow * 2^B / denom) mod 2^B,  pow = start_bit + p.
 *
 * When pow >= 0, write 2^pow = q*denom + r, where
 * r = 2^pow mod denom.  The q*2^B part vanishes modulo 2^B, leaving
 *
 *   floor(r * 2^B / denom),
 *
 * so pow_mod followed by one shift and division is sufficient.  When
 * -B <= pow < 0, the contribution is directly
 * floor(2^(B+pow)/denom); when pow < -B it is below the retained
 * precision and is zero.  The calculation keeps 64 guard bits and drops
 * them at the end, yielding the requested 256 fractional bits of pi.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <bigint.hpp>
#include <cucheck.hpp>

using namespace cupqc;
using namespace std;

// SM<> selects the architecture the descriptor generates code for, and defaults
// to SM<800> when omitted. Deriving it from __CUDA_ARCH__ keeps it in step with
// the Makefile's -arch=native. Architectures with no SM<> specialization fall
// through to the default.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define PIHEX_SM() + SM<900>()
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 890)
#define PIHEX_SM() + SM<890>()
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 870)
#define PIHEX_SM() + SM<870>()
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
#define PIHEX_SM() + SM<860>()
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
#define PIHEX_SM() + SM<800>()
#else
#define PIHEX_SM()
#endif

using bigint64_cfg = decltype(BitWidth<64>() + Thread() PIHEX_SM());
using bigint320_cfg = decltype(BitWidth<320>() + Thread() PIHEX_SM());
using bigint384_cfg = decltype(BitWidth<384>() + Thread() PIHEX_SM());

using bigint64_t = typename bigint64_cfg::bigint;
using bigint320_t = typename bigint320_cfg::bigint;
using bigint384_t = typename bigint384_cfg::bigint;

constexpr int BIGINT_WORDS = bigint320_cfg::num_limbs;
// Reserve 64 guard bits because denominators are limited to 64 bits.
// Since max_denom is approximately start_bit + 338, this corresponds
// mathematically to start_bit <~ 2^64. The current int64_t indexing
// imposes a lower practical limit of approximately 2^63.
constexpr int GUARD_WORDS = bigint64_cfg::num_limbs;
constexpr int RESULT_WORDS = BIGINT_WORDS - GUARD_WORDS;
static_assert(bigint384_cfg::num_limbs == BIGINT_WORDS + GUARD_WORDS);

constexpr int THREADS_PER_BLOCK = 64;
constexpr int REDUCE_THREADS = 1024;
constexpr int BELLARD_TERM_PAIRS = 7;

// Explicitly narrow to the low 320 bits.
__device__ bigint320_t narrow_to_320(const bigint384_t& value)
{
    bigint320_t result;
    for (int i = 0; i < BIGINT_WORDS; ++i)
        result[i] = value[i];
    return result;
}

// term = floor(2^pow * 2^320 / denom) mod 2^320.
// The denominator is deliberately always represented by a 64-bit bigint.
__device__ void pihex_term(bigint320_t& term, int64_t pow, uint64_t denom)
{
    bigint64_t divisor(denom);
    bigint384_t quotient;
    bigint64_t remainder;

    if (pow > 0) {
        bigint64_t residue = bigint64_t(2u).pow_mod(static_cast<uint64_t>(pow), divisor);
        bigint384_t dividend(residue.to_uint64());
        dividend <<= 320;
        dividend.div_rem(divisor, quotient, remainder);
        term = narrow_to_320(quotient);
    } else if (pow >= -320) {
        // Negative powers occur in the tail (and at small start_bit values).
        bigint384_t dividend(1u);
        dividend <<= static_cast<uint32_t>(320 + pow);
        dividend.div_rem(divisor, quotient, remainder);
        term = narrow_to_320(quotient);
    } else {
        term = bigint320_t{};
    }
}

__device__ void accum_pihex_terms(bigint320_t& accum,
                                  int64_t pow_a, uint64_t denom_a,
                                  int64_t pow_b, uint64_t denom_b)
{
    bigint320_t term_a;
    bigint320_t term_b;
    pihex_term(term_a, pow_a, denom_a);
    pihex_term(term_b, pow_b, denom_b);
    accum += term_a;
    accum -= term_b;
}

__device__ void accum_bellard_pair(bigint320_t& accum, int term_id,
                                   int64_t start_bit, int64_t n)
{
    switch (term_id) {
    case 0:
        accum_pihex_terms(accum, start_bit-20*n-16+5, 8*n+5,
                          start_bit-20*n-6+5, 8*n+1);
        break;
    case 1:
        accum_pihex_terms(accum, start_bit-20*n-16, 8*n+7,
                          start_bit-20*n-6, 8*n+3);
        break;
    case 2:
        accum_pihex_terms(accum, start_bit-20*n-6+8, 20*n+1,
                          start_bit-20*n-16+8, 20*n+11);
        break;
    case 3:
        accum_pihex_terms(accum, start_bit-20*n-16+6, 20*n+13,
                          start_bit-20*n-6+6, 20*n+3);
        break;
    case 4:
        accum_pihex_terms(accum, start_bit-20*n-16+2, 20*n+15,
                          start_bit-20*n-6+2, 20*n+5);
        break;
    case 5:
        accum_pihex_terms(accum, start_bit-20*n-16+2, 20*n+17,
                          start_bit-20*n-6+2, 20*n+7);
        break;
    default:
        accum_pihex_terms(accum, start_bit-20*n-6, 20*n+9,
                          start_bit-20*n-16, 20*n+19);
        break;
    }
}

__global__ void calc_pihex(uint32_t* partials, int64_t start_bit,
                           int64_t n_max, int blocks_per_term)
{
    const int term_id = blockIdx.x / blocks_per_term;
    const int local_block = blockIdx.x % blocks_per_term;
    const int64_t tid = static_cast<int64_t>(local_block) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blocks_per_term) * blockDim.x;

    bigint320_t accum;
    for (int64_t n = tid; n < n_max; n += stride) {
        accum_bellard_pair(accum, term_id, start_bit, n);
    }

    bigint320_t other;
    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int limb = 0; limb < BIGINT_WORDS; ++limb) {
            other[limb] = __shfl_down_sync(0xffffffffu, accum[limb], offset);
        }
        accum += other;
    }

    if ((threadIdx.x & 31) == 0) {
        const int partial = blockIdx.x * (THREADS_PER_BLOCK / 32) + threadIdx.x / 32;
        accum.store(partials, partial);
    }
}

__global__ void reduce_pihex(uint32_t* partials, int num_partials)
{
    __shared__ uint32_t shared[REDUCE_THREADS * BIGINT_WORDS];
    const int tid = threadIdx.x;
    bigint320_t accum;

    for (int i = tid; i < num_partials; i += REDUCE_THREADS) {
        bigint320_t value(partials, i);
        accum += value;
    }
    accum.store(shared, tid);
    __syncthreads();

    for (int stride = REDUCE_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            bigint320_t a(shared, tid);
            bigint320_t b(shared, tid + stride);
            a += b;
            a.store(shared, tid);
        }
        __syncthreads();
    }

    if (tid == 0) {
        bigint320_t result(shared, 0);
        result.store(partials);
    }
}

// Known leading hex digits of pi (after the leading "3."), used to validate
// the start_bit=0 run.  Stored in cuPQC limb order (LSW first); printed in
// reverse order to recover the natural human-readable hex string
// "243f6a8885a308d3...082efa98ec4e6c89".
constexpr uint32_t PI_AT_ZERO[RESULT_WORDS] = {
    0xec4e6c89u, 0x082efa98u, 0x299f31d0u, 0xa4093822u,
    0x03707344u, 0x13198a2eu, 0x85a308d3u, 0x243f6a88u,
};

int run_pihex(int64_t start_bit)
{
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int64_t n_max = (start_bit + 320 + 19) / 20;
    const int64_t wanted_blocks = (n_max + THREADS_PER_BLOCK - 1) /
                                  THREADS_PER_BLOCK;
    const int64_t max_blocks = static_cast<int64_t>(prop.multiProcessorCount) * 8;
    const int blocks_per_term = static_cast<int>(min(wanted_blocks, max_blocks));
    const int num_partials = blocks_per_term * BELLARD_TERM_PAIRS *
                             (THREADS_PER_BLOCK / 32);

    uint32_t* partials = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&partials),
                          static_cast<size_t>(num_partials) * BIGINT_WORDS * sizeof(uint32_t)));

    array<uint32_t, RESULT_WORDS> result{};
    CUDA_CHECK(cudaHostRegister(result.data(), result.size() * sizeof(uint32_t),
                                cudaHostRegisterDefault));
    cudaStream_t stream{};
    cudaEvent_t begin{}, calculated{}, reduced{}, copied{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&calculated));
    CUDA_CHECK(cudaEventCreate(&reduced));
    CUDA_CHECK(cudaEventCreate(&copied));
    CUDA_CHECK(cudaEventRecord(begin, stream));

    calc_pihex<<<blocks_per_term * BELLARD_TERM_PAIRS, THREADS_PER_BLOCK, 0, stream>>>(
        partials, start_bit, n_max, blocks_per_term);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(calculated, stream));
    reduce_pihex<<<1, REDUCE_THREADS, 0, stream>>>(partials, num_partials);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(reduced, stream));

    CUDA_CHECK(cudaMemcpyAsync(result.data(), partials + GUARD_WORDS,
                               result.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(copied, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaHostUnregister(result.data()));

    float calc_ms = 0.0f;
    float reduce_ms = 0.0f;
    float memcpy_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&calc_ms, begin, calculated));
    CUDA_CHECK(cudaEventElapsedTime(&reduce_ms, calculated, reduced));
    CUDA_CHECK(cudaEventElapsedTime(&memcpy_ms, reduced, copied));

    printf("pihex Example Program Results\n");
    printf("pihex start_bit=%lld GPU timings: calc=%.3fms reduce=%.3fms memcpy=%.3fms\n",
           static_cast<long long>(start_bit), calc_ms, reduce_ms, memcpy_ms);
    printf("pihex start_bit=%lld (64 hex digits, most-significant first):\n  ",
           static_cast<long long>(start_bit));
    for (int i = RESULT_WORDS - 1; i >= 0; --i) {
        printf("%08x", result[i]);
    }
    printf("\n");

    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(calculated));
    CUDA_CHECK(cudaEventDestroy(reduced));
    CUDA_CHECK(cudaEventDestroy(copied));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(partials));

    if (start_bit == 0 && !equal(result.begin(), result.end(), PI_AT_ZERO)) {
        fprintf(stderr, "FAIL: result does not match the leading hex digits of pi.\n");
        return EXIT_FAILURE;
    }
    if (start_bit == 0) {
        printf("OK: matches the known leading hex digits of pi.\n");
    }
    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    int64_t start_bit = 0;
    if (argc > 1) {
        char* end = nullptr;
        const long long parsed = strtoll(argv[1], &end, 0);
        if (end == argv[1] || *end != '\0' || parsed < 0) {
            fprintf(stderr, "Usage: %s [nonnegative start_bit]\n", argv[0]);
            return EXIT_FAILURE;
        }
        start_bit = static_cast<int64_t>(parsed);
    }

    printf("================================================================\n");
    printf("BigInt pihex Example\n");
    printf("================================================================\n\n");
    printf("This example computes 256 fractional hex digits of pi starting at\n");
    printf("bit offset %lld, using Bellard's formula evaluated in 320- and\n",
           static_cast<long long>(start_bit));
    printf("384-bit integer arithmetic across the whole GPU. It combines\n");
    printf("pow_mod, div_rem, shifts and warp-level reduction of big integers.\n\n");
    printf("Configuration: 320-bit accumulators, thread execution, 64 guard bits\n\n");

    return run_pihex(start_bit);
}
