/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>
#include <iomanip>
#include <iostream>

#include <hash.hpp>

using namespace cupqc;

// Use KoalaBear field for this example
using HASH = decltype(POSEIDON2_KB_8_16() + Thread());

__global__ void hash_poseidon2_kernel(uint32_t* digest, const uint32_t* msg, size_t inbuf_len, size_t out_len)
{
    // Poseidon2 with Capacity 8 and Width 16 with KoalaBear field
    HASH hash {};
    hash.reset();
    hash.update(msg, inbuf_len);
    hash.finalize();
    hash.digest(digest, out_len);
}

void hash_poseidon2(std::vector<uint32_t>& digest, std::vector<uint32_t>& msg, size_t out_len)
{
    uint32_t* d_msg;
    uint32_t* d_digest;
    cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size() * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_digest), digest.size() * sizeof(uint32_t));

    cudaMemcpy(d_msg, msg.data(), msg.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Note: Poseidon2 hash function uses thread configuration, so we use a single thread to hash a single message.
    hash_poseidon2_kernel<<<1, 1>>>(d_digest, d_msg, msg.size(), out_len);

    cudaMemcpy(digest.data(), d_digest, digest.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_msg);
    cudaFree(d_digest);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "================================================================\n";
    std::cout << "Poseidon2 Hash Function Example\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "This example demonstrates Poseidon2 hashing using cuPQC SDK.\n";
    std::cout << "Poseidon2 is a zero-knowledge-friendly hash function designed for\n";
    std::cout << "efficient use in zero-knowledge proof systems. It uses arithmetic\n";
    std::cout << "operations native to ZK circuits.\n\n";
    std::cout << "Configuration: Capacity 8, Width 16, KoalaBear field\n\n";
    
    constexpr size_t in_len = 64;
    constexpr size_t out_len = 16;
    std::vector<uint32_t> msg(in_len, 0);

    // Generate sample input data (field elements)
    for (size_t i = 0; i < in_len; i++) {
        msg[i] = i % cupqc_common::BabyBearPrime;
    }
    
    std::cout << "========================================\n";
    std::cout << "Input Data\n";
    std::cout << "========================================\n";
    std::cout << "Input size: " << in_len << " field elements (" << (in_len * sizeof(uint32_t)) << " bytes)\n";
    std::cout << "Input data (first 8 values):\n";
    for (size_t i = 0; i < std::min(in_len, size_t(8)); i++) {
        std::cout << "  Input[" << i << "]: 0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << msg[i] << std::dec << " (" << msg[i] << ")\n";
    }
    if (in_len > 8) {
        std::cout << "  ... (showing first 8 of " << in_len << " values)\n";
    }
    std::cout << "\n";
    
    // Compute hash
    std::cout << "========================================\n";
    std::cout << "Computing Poseidon2 Hash\n";
    std::cout << "========================================\n";
    std::vector<uint32_t> digest(out_len, 0);
    hash_poseidon2(digest, msg, out_len);
    
    // Display results
    std::cout << "Computed Hash (" << out_len << " field elements):\n";
    std::cout << "  ";
    for (size_t i = 0; i < digest.size(); i++) {
        std::cout << "0x" << std::hex << std::setw(8) << std::setfill('0') << digest[i] << std::dec;
        if (i < digest.size() - 1) {
            std::cout << " ";
        }
    }
    std::cout << "\n\n";
    
    std::cout << "Hash values (decimal):\n";
    std::cout << "  ";
    for (size_t i = 0; i < digest.size(); i++) {
        std::cout << digest[i];
        if (i < digest.size() - 1) {
            std::cout << " ";
        }
    }
    std::cout << "\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Hash Information\n";
    std::cout << "========================================\n";
    std::cout << "Hash size: " << out_len << " field elements (" << (out_len * sizeof(uint32_t)) << " bytes)\n";
    std::cout << "Field: KoalaBear\n";
    std::cout << "Poseidon2 parameters: Capacity=8, Width=16\n\n";
    
    std::cout << "Example completed successfully.\n";
    return 0;
}
