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
#include <chrono>

#include <hash.hpp>

using namespace cupqc;

using HASH   = decltype(POSEIDON2_BB_8_16() + Thread());
using MERKLE = decltype(MERKLE_FIELD_2097152() + BlockDim<256>());
using SUB_MERKLE = decltype(MERKLE_FIELD_2048() + BlockDim<256>());
using FINAL_MERKLE = decltype(MERKLE_FIELD_1024() + BlockDim<256>());

__global__ void create_leaves_kernel(tree<MERKLE::Size, HASH, uint32_t> merkle, const uint32_t* msg, size_t inbuf_len)
{
    HASH hash{};
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < MERKLE::Size) {
            MERKLE().create_leaf(merkle.nodes + i * merkle.digest_size, msg + i * inbuf_len, hash, inbuf_len);
    }
}

__global__ void generate_sub_tree_kernel(tree<MERKLE::Size, HASH, uint32_t> merkle)
{
    HASH hash{};
    SUB_MERKLE().generate_sub_tree(hash, merkle, blockIdx.x);
}

__global__ void generate_final_tree_kernel(tree<MERKLE::Size, HASH, uint32_t> merkle, uint32_t* root)
{
    HASH hash;
    tree<FINAL_MERKLE::Size, HASH, uint32_t> merkle_final;
    const size_t left_overs = merkle.size - merkle_final.size;
    merkle_final.nodes      = merkle.nodes + left_overs * merkle.digest_size;
    
   __syncthreads();
   FINAL_MERKLE().generate_tree(hash, merkle_final);

   if(threadIdx.x == 0) {
       for(uint32_t i = 0; i < merkle.digest_size; i++) {
           root[i] = merkle.root()[i];
       }
   }
}

void generate_tree(const std::vector<uint32_t>& msg, size_t inbuf_len, tree<MERKLE::Size, HASH, uint32_t> merkle, uint32_t* d_root)
{
    uint32_t* d_msg;
    cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size() * sizeof(uint32_t)); //Here msg.size has the number of messages and the length of each message is inbuf_len
    cudaMemcpy(d_msg, msg.data(), msg.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    create_leaves_kernel<<<MERKLE::Size / MERKLE::BlockDim.x, MERKLE::BlockDim>>>(merkle, d_msg, inbuf_len);
    generate_sub_tree_kernel<<<FINAL_MERKLE::Size , MERKLE::BlockDim>>>(merkle);
    generate_final_tree_kernel<<<1, MERKLE::BlockDim>>>(merkle, d_root);
    cudaFree(d_msg); //Device side message is no longer needed
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "================================================================\n";
    std::cout << "Multi-Block Merkle Tree Generation Example\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "This example demonstrates generating a very large Merkle tree using\n";
    std::cout << "multiple GPU blocks in parallel. The multi-block approach divides the\n";
    std::cout << "tree into sub-trees that are generated in parallel across multiple\n";
    std::cout << "GPU blocks, then combines them into a final tree. This enables\n";
    std::cout << "efficient generation of extremely large trees (millions of leaves).\n\n";
    
    constexpr auto N = MERKLE::Size;
    constexpr size_t in_len = 64;
    
    std::cout << "Tree Configuration:\n";
    std::cout << "  Total leaves: " << N << " (" << (N / 1000000.0) << " million)\n";
    std::cout << "  Leaf size: " << in_len << " field elements (" << (in_len * sizeof(uint32_t)) << " bytes)\n";
    std::cout << "  Total data: " << (in_len * N * sizeof(uint32_t) / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Sub-tree size: " << SUB_MERKLE::Size << " leaves\n";
    std::cout << "  Number of sub-trees combined in final tree: " << FINAL_MERKLE::Size << "\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Generating Large Merkle Tree\n";
    std::cout << "========================================\n";
    std::cout << "Generating input data...\n";
    
    std::vector<uint32_t> msg(in_len * N, 0);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < in_len; j++) {
            msg[i * in_len + j] = (i * j) % cupqc_common::BabyBearPrime;
        }
    }
    std::cout << "Input data generated.\n\n";
    
    std::cout << "Allocating tree on GPU...\n";
    tree<MERKLE::Size, HASH, uint32_t> merkle;
    merkle.allocate_tree();
    uint32_t* d_root = nullptr;
    uint32_t* root = new uint32_t[merkle.digest_size];
    cudaMalloc(reinterpret_cast<void**>(&d_root), merkle.digest_size * sizeof(uint32_t));
    
    std::cout << "Generating Merkle tree using multi-block approach...\n";
    std::cout << "  Step 1: Creating " << N << " leaves in parallel\n";
    std::cout << "  Step 2: Generating " << FINAL_MERKLE::Size << " sub-trees in parallel (multi-block)\n";
    std::cout << "  Step 3: Combining sub-trees into final tree\n";
    
    // Time the tree generation
    auto start_time = std::chrono::high_resolution_clock::now();
    generate_tree(msg, in_len, merkle, d_root);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    cudaMemcpy(root, d_root, merkle.digest_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    std::cout << "\nMerkle tree generated successfully!\n";
    std::cout << "  Generation time: " << duration.count() << " ms (" << (duration.count() / 1000.0) << " seconds)\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
              << (N / (duration.count() / 1000.0)) << " leaves/second\n\n";
    
    std::cout << "========================================\n";
    std::cout << "Tree Root Hash\n";
    std::cout << "========================================\n";
    for(uint32_t i = 0; i < merkle.digest_size; i++) {
        std::cout << "  Root[" << i << "]: 0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << root[i] << std::dec << "\n";
    }
    std::cout << "\n";
    
    std::cout << "Successfully generated " << N << " leaf Merkle tree using multi-block GPU parallelization.\n\n";
    
    // Cleanup
    merkle.free_tree();
    cudaFree(d_root);
    delete[] root;
    
    std::cout << "Example completed successfully.\n";
    return 0;
}
