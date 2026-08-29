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

// cuPQC BigInt Add / ModMul Example
//
// A minimal, single-thread demonstration of two of the most common
// cuPQC-BigInt operations: plain addition and modular multiplication.

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

// 256-bit integers, one thread per value.
using BI256 = decltype(BitWidth<256>() + Thread() BIGINT_EXAMPLE_SM());

__global__ void add_mulmod_kernel(uint32_t* sums, uint32_t* products,
                                   const uint32_t* global_buf_a,
                                   const uint32_t* global_buf_b,
                                   const uint32_t* global_buf_m,
                                   unsigned int count)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const BI256::bigint a(global_buf_a, index);
    const BI256::bigint b(global_buf_b, index);
    const BI256::bigint m(global_buf_m, index);

    // 256-bit integer addition (wraps on overflow).
    const auto sum = a + b;
    // (a * b) mod m.
    const auto product = a.mul_mod(b, m);

    sum.store(sums, index);
    product.store(products, index);
}

int main()
{
    std::printf("================================================================\n");
    std::printf("BigInt Addition and Modular Multiplication Example\n");
    std::printf("================================================================\n\n");

    std::printf("This example demonstrates two of the most common cuPQC-BigInt\n");
    std::printf("operations: fixed-width addition, which wraps modulo 2^256, and\n");
    std::printf("mul_mod, which computes the full double-width product and then\n");
    std::printf("reduces it modulo m.\n\n");
    std::printf("Configuration: 256-bit width, thread execution (one value per thread)\n");
    std::printf("Limb layout:   %u little-endian 32-bit limbs per value\n\n",
                static_cast<unsigned int>(BI256::num_limbs));

    // Small values keep host-side verification easy; the same layout
    // supports arbitrary 256-bit input values.
    const unsigned int count = 2;
    using host_limb_array = std::array<uint32_t, BI256::num_limbs>;
    static_assert(sizeof(host_limb_array) == BI256::num_limbs * sizeof(uint32_t));
    std::vector<host_limb_array> a(count, host_limb_array{});
    std::vector<host_limb_array> b(count, host_limb_array{});
    std::vector<host_limb_array> m(count, host_limb_array{});
    std::vector<host_limb_array> sums(count);
    std::vector<host_limb_array> products(count);

    a[0][0] = 7u;
    b[0][0] = 5u;
    m[0][0] = 11u;   // (7 + 5) = 12, (7 * 5) mod 11 = 2
    a[1][0] = 40u;
    b[1][0] = 9u;
    m[1][0] = 13u;   // (40 + 9) = 49, (40 * 9) mod 13 = 3

    uint32_t *d_a = nullptr, *d_b = nullptr, *d_m = nullptr;
    uint32_t *d_sums = nullptr, *d_products = nullptr;
    const size_t bytes = a.size() * sizeof(host_limb_array);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_m), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sums), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_products), bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, m.data(), bytes, cudaMemcpyHostToDevice));

    constexpr unsigned int kThreads = 128;
    add_mulmod_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
        d_sums, d_products, d_a, d_b, d_m, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(sums.data(), d_sums, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(products.data(), d_products, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_products));

    std::printf("========================================\n");
    std::printf("Results\n");
    std::printf("========================================\n");
    std::printf("  7 + 5            = %u (expected 12)\n", sums[0][0]);
    std::printf("  (7 * 5)  mod 11  = %u (expected 2)\n", products[0][0]);
    std::printf("  40 + 9           = %u (expected 49)\n", sums[1][0]);
    std::printf("  (40 * 9) mod 13  = %u (expected 3)\n\n", products[1][0]);

    if (sums[0][0] != 12u || products[0][0] != 2u ||
        sums[1][0] != 49u || products[1][0] != 3u) {
        std::fprintf(stderr, "FAIL: unexpected sums or products.\n");
        return EXIT_FAILURE;
    }

    std::printf("Example completed successfully.\n");
    return EXIT_SUCCESS;
}
