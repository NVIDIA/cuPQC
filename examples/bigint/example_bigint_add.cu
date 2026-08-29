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

// cuPQC BigInt VecAdd Example
//
// Vector addition is the "hello world" of CUDA kernel programming.
//
// This kernel adds two vectors of 1024-bit integers, with each addition using
// 4 threads cooperating per integer operation (TPI=4).

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <bigint.hpp>
#include <cucheck.hpp>

using namespace cupqc;

// SM<> selects the architecture the descriptor generates code for, and defaults
// to SM<800> when omitted. Deriving it from __CUDA_ARCH__ keeps it in step with
// the Makefile's -arch=native. Architectures with no SM<> specialization fall
// through to the default.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
#define BIGINT_EXAMPLE_SM() + SM<900>()
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 890)
#define BIGINT_EXAMPLE_SM() + SM<890>()
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 870)
#define BIGINT_EXAMPLE_SM() + SM<870>()
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
#define BIGINT_EXAMPLE_SM() + SM<860>()
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
#define BIGINT_EXAMPLE_SM() + SM<800>()
#else
#define BIGINT_EXAMPLE_SM()
#endif

// 1024-bit integers, with four threads cooperating on each integer.
using BI1024 = decltype(BitWidth<1024>() + TPI<4>() + Warp() BIGINT_EXAMPLE_SM());

__global__ void add_kernel(uint32_t* sums,
                           const uint32_t* global_buf_a,
                           const uint32_t* global_buf_b,
                           unsigned int count)
{
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // each group of 4 threads owns one 1024-bit integer
    const unsigned int bigint_index = thread_id / BI1024::tpi;
    if (bigint_index >= count) {
        return;
    }

    // The indexed constructor reads the integer from the packed limb buffer
    const BI1024::bigint a(global_buf_a, bigint_index);
    const BI1024::bigint b(global_buf_b, bigint_index);
    // 1024-bit integer addition (wraps on overflow)
    const auto c = a + b;
    // store the result in the packed output limb buffer
    c.store(sums, bigint_index);
}

int main()
{
    std::printf("================================================================\n");
    std::printf("BigInt Vector Addition Example\n");
    std::printf("================================================================\n\n");

    std::printf("This example demonstrates 1024-bit integer addition using cuPQC SDK.\n");
    std::printf("cuPQC-BigInt is a device-side library: every operation is called from\n");
    std::printf("inside your own kernel rather than launched from the host.\n\n");
    std::printf("Configuration: 1024-bit width, warp execution, %u threads per instance\n",
                static_cast<unsigned int>(BI1024::tpi));
    std::printf("Limb layout:   %u little-endian 32-bit limbs per value\n\n",
                static_cast<unsigned int>(BI1024::num_limbs));

    // Two small values make host-side verification easy; the same layout
    // supports arbitrary 1024-bit input values.
    const unsigned int count = 2;
    using host_limb_array = std::array<uint32_t, BI1024::num_limbs>;
    static_assert(sizeof(host_limb_array) == BI1024::num_limbs * sizeof(uint32_t));
    std::vector<host_limb_array> a(count);
    std::vector<host_limb_array> b(count);
    std::vector<host_limb_array> sums(count);
    a[0][0] = 2u;
    a[1][0] = 10u;
    b[0][0] = 3u;
    b[1][0] = 20u;

    uint32_t *d_a = nullptr, *d_b = nullptr, *d_sums = nullptr;
    const size_t bytes = a.size() * sizeof(host_limb_array);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sums), bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    constexpr unsigned int kThreads = 128;
    add_kernel<<<(count * BI1024::tpi + kThreads - 1) / kThreads, kThreads>>>(
        d_sums, d_a, d_b, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(sums.data(), d_sums, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_sums));

    std::printf("========================================\n");
    std::printf("Results\n");
    std::printf("========================================\n");
    std::printf("  2 + 3   = %u (expected 5)\n", sums[0][0]);
    std::printf("  10 + 20 = %u (expected 30)\n\n", sums[1][0]);

    if (sums[0][0] != 5u || sums[1][0] != 30u) {
        std::fprintf(stderr, "FAIL: unexpected sums.\n");
        return EXIT_FAILURE;
    }

    std::printf("Example completed successfully.\n");
    return EXIT_SUCCESS;
}
