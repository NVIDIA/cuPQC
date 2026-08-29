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
#include <cstring>
#include <hash.hpp>

using namespace cupqc;

using SHA2_256_THREAD = decltype(SHA2_256() + Thread());

__global__ void hash_sha2_kernel(uint8_t* digest, const uint8_t* msg, size_t inbuf_len)
{
    SHA2_256_THREAD hash {};
    if (threadIdx.x == 0) {
        hash.reset();
        hash.update(msg, inbuf_len);
        hash.finalize();
        hash.digest(digest, SHA2_256_THREAD::digest_size);
    }
}


void hash_sha2(std::vector<uint8_t>& digest, std::vector<uint8_t>& msg)
{
    uint8_t* d_msg;
    uint8_t* d_digest;
    cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size());
    cudaMalloc(reinterpret_cast<void**>(&d_digest), digest.size());

    cudaMemcpy(d_msg, msg.data(), msg.size(), cudaMemcpyHostToDevice);

    hash_sha2_kernel<<<1, 32>>>(d_digest, d_msg, msg.size());

    cudaMemcpy(digest.data(), d_digest, digest.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_msg);
    cudaFree(d_digest);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "================================================================\n";
    std::cout << "SHA-2 (SHA-256) Hash Function Example\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "This example demonstrates SHA-256 hashing using cuPQC SDK.\n";
    std::cout << "SHA-256 is a cryptographic hash function that produces a 256-bit\n";
    std::cout << "(32-byte) hash value. It is widely used for data integrity verification.\n\n";
    
    // Input message
    const char * msg_str = "The quick brown fox jumps over the lazy dog";
    std::vector<uint8_t> msg(reinterpret_cast<const uint8_t*>(msg_str), 
                             reinterpret_cast<const uint8_t*>(msg_str) + strlen(msg_str));
    
    std::cout << "========================================\n";
    std::cout << "Input Data\n";
    std::cout << "========================================\n";
    std::cout << "Input text: \"" << msg_str << "\"\n";
    std::cout << "Input size: " << msg.size() << " bytes\n\n";
    
    // Known expected hash for "The quick brown fox jumps over the lazy dog"
    // This is the standard test vector for SHA-256
    const char* expected_hash_str = "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592";
    std::vector<uint8_t> expected_digest(SHA2_256::digest_size, 0);
    
    // Convert expected hash string to bytes
    for (size_t i = 0; i < expected_digest.size(); i++) {
        std::string byte_str = std::string(1, expected_hash_str[i*2]) + expected_hash_str[i*2+1];
        expected_digest[i] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
    }
    
    // Compute hash
    std::cout << "========================================\n";
    std::cout << "Computing SHA-256 Hash\n";
    std::cout << "========================================\n";
    std::vector<uint8_t> digest(SHA2_256::digest_size, 0);
    hash_sha2(digest, msg);
    
    // Display results
    std::cout << "Computed Hash:\n";
    std::cout << "  ";
    for (uint8_t num : digest) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(num);
    }
    std::cout << std::dec << "\n\n";
    
    std::cout << "Expected Hash (precomputed):\n";
    std::cout << "  ";
    for (uint8_t num : expected_digest) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(num);
    }
    std::cout << std::dec << "\n\n";
    
    // Compare
    std::cout << "========================================\n";
    std::cout << "Hash Verification\n";
    std::cout << "========================================\n";
    bool match = true;
    for (size_t i = 0; i < digest.size(); i++) {
        if (digest[i] != expected_digest[i]) {
            match = false;
            break;
        }
    }
    
    if (match) {
        std::cout << "✓ Hash verification: VALID\n";
        std::cout << "  The computed hash matches the expected hash.\n";
        std::cout << "  SHA-256 implementation is working correctly.\n";
    } else {
        std::cout << "✗ Hash verification: INVALID\n";
        std::cout << "  The computed hash does not match the expected hash.\n";
    }
    std::cout << "\n";
    
    std::cout << "Example completed successfully.\n";
    return 0;
}
