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
#include <random>

#include <hash.hpp>

using namespace cupqc;

using HASH   = decltype(POSEIDON2_BB_8_16() + Thread());

//Will use MERKLE_FIELD_2048 in this example, we require FIELD when using the Poseidon2 hash function, and we use 256 threads per block
using MERKLE = decltype(MERKLE_FIELD_2048() + BlockDim<256>());

template<class Merkle, class Hash, typename Precision>
__global__ void generate_merkle_tree_kernel(tree<Merkle::Size, Hash, Precision> merkle, const Precision* msg, size_t inbuf_len, Precision* d_root)
{
    Hash hash{};
    for(uint32_t i = threadIdx.x; i < Merkle::Size; i += blockDim.x) {
        Merkle().create_leaf(merkle.nodes + i * merkle.digest_size, msg + i * inbuf_len, hash, inbuf_len);
    }

    Merkle().generate_tree(hash, merkle);
    if(threadIdx.x == 0) {
        for(uint32_t i = 0; i < merkle.digest_size; i++)
        d_root[i] = merkle.root()[i];
    }
}

template<class Merkle, class Hash, typename Precision>
__global__ void single_proof_kernel(proof<Merkle::Size, Hash, Precision> this_proof, const Precision* leaf_to_prove, const uint32_t leaf_index, const tree<Merkle::Size, Hash, Precision> merkle)
{
    Merkle().generate_proof(this_proof, leaf_to_prove, leaf_index, merkle);
}


template<class Merkle, class Hash, typename Precision>
__global__ void generate_proof_kernel(proof<Merkle::Size, Hash, Precision>* proofs, const Precision* leaves_to_prove, const uint32_t* leaf_indices, const tree<Merkle::Size, Hash, Precision> merkle)
{
    constexpr size_t digest_size = proof<Merkle::Size, Hash, Precision>::digest_size;
    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const Precision* leaf_to_prove = leaves_to_prove + thread_index * digest_size;
    const uint32_t leaf_index = leaf_indices[thread_index];
    Merkle().generate_proof(proofs[thread_index], leaf_to_prove, leaf_index, merkle); // Each thread generates a proof for the given leaf and corresponding index
}

template<class Merkle, class Hash, typename Precision>
__global__ void single_verify_kernel(const proof<Merkle::Size, Hash, Precision> this_proof, const Precision* verify_leaf, const uint32_t verify_index, const Precision* root, bool* verified) 
{
    Hash hash{};
    *verified = Merkle().verify_proof(this_proof, verify_leaf, verify_index, root, hash);
}


template<class Merkle, class Hash, typename Precision>
__global__ void verify_proof_kernel(const proof<Merkle::Size, Hash, Precision>* proofs, const Precision* verify_leaves, const uint32_t* verify_indices, const Precision* root, bool* verified) 
{
    Hash hash{};
    constexpr size_t digest_size = proof<Merkle::Size, Hash, Precision>::digest_size;
    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t verify_index = verify_indices[thread_index]; // This is the leaf's index in the tree
    const Precision* verify_leaf = verify_leaves + thread_index * digest_size;
    verified[thread_index] = Merkle().verify_proof(proofs[thread_index], verify_leaf, verify_index, root, hash);
}

template<class Merkle, class Hash, typename Precision>
void generate_tree(const std::vector<Precision>& msg, size_t inbuf_len, tree<Merkle::Size, Hash, Precision> merkle, Precision* d_root)
{
    Precision* d_msg;
    cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size() * sizeof(Precision)); //Here msg.size has the number of messages and the length of each message is inbuf_len
    cudaMemcpy(d_msg, msg.data(), msg.size() * sizeof(Precision), cudaMemcpyHostToDevice);
    generate_merkle_tree_kernel<Merkle, Hash, Precision><<<1, Merkle::BlockDim>>>(merkle, d_msg, inbuf_len, d_root);
    cudaFree(d_msg); //Device side message is no longer needed
}

template<class Merkle, class Hash, typename Precision>
void generate_proof(const Precision* leaves_to_prove, const uint32_t* indices_to_prove, const size_t num_proofs, proof<Merkle::Size, Hash, Precision>* proofs, 
                    const tree<Merkle::Size, Hash, Precision> merkle)
{
    dim3 gridDim((num_proofs + 255) / 256);
    dim3 blockDim(256);
    if(num_proofs < 256) {
        gridDim.x = 1;
        blockDim.x = num_proofs;
    }
    generate_proof_kernel<Merkle, Hash, Precision><<<gridDim, blockDim>>>(proofs, leaves_to_prove, indices_to_prove, merkle);
}

template<class Merkle, class Hash, typename Precision>
void verify_proof(const Precision* verify_leaves, const uint32_t* verify_indices, const size_t num_proofs, const proof<Merkle::Size, Hash, Precision>* proofs, 
                  const Precision* root, bool* verified)
{
    dim3 gridDim((num_proofs + 255) / 256);
    dim3 blockDim(256);
    if(num_proofs < 256) {
        gridDim.x = 1;
        blockDim.x = num_proofs;
    }
    verify_proof_kernel<Merkle, Hash, Precision><<<gridDim, blockDim>>>(proofs, verify_leaves, verify_indices, root, verified);
}

//This is simply a helper kernel to set the indices for this example.
__global__ void index_set(uint32_t* indices, size_t num_indices) {
    for (size_t i = threadIdx.x; i < num_indices; i += blockDim.x) {
        indices[i] = i;
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "================================================================\n";
    std::cout << "Merkle Tree Proof Generation and Verification Example\n";
    std::cout << "================================================================\n\n";
    
    std::cout << "This example demonstrates Merkle tree construction, proof generation,\n";
    std::cout << "and verification using cuPQC SDK. Merkle trees enable efficient\n";
    std::cout << "membership proofs - you can prove a leaf belongs to the tree without\n";
    std::cout << "revealing other leaves. This is essential for zero-knowledge proofs\n";
    std::cout << "and data integrity verification.\n\n";
    std::cout << "Configuration: " << MERKLE::Size << " leaves, Poseidon2 hash (BabyBear field)\n\n";

    using tree_type = tree<MERKLE::Size, HASH, uint32_t>;
    using proof_type = proof<MERKLE::Size, HASH, uint32_t>;
    constexpr auto N = MERKLE::Size;

    constexpr size_t in_len = 64;
    std::vector<uint32_t> msg(in_len * N, 0);

    // Generate random input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, cupqc_common::BabyBearPrime - 1);
    
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < in_len; j++) {
            msg[i * in_len + j] = dis(gen);
        }
    }
    
    std::cout << "========================================\n";
    std::cout << "Input Data\n";
    std::cout << "========================================\n";
    std::cout << "Number of messages: " << N << "\n";
    std::cout << "Message size: " << in_len << " field elements (" << (in_len * sizeof(uint32_t)) << " bytes)\n";
    std::cout << "Total input size: " << (in_len * N * sizeof(uint32_t)) << " bytes\n";
    std::cout << "Sample message 0 (first 4 values):\n";
    for (size_t j = 0; j < std::min(in_len, size_t(4)); j++) {
        std::cout << "  Msg[0][" << j << "]: 0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << msg[j] << std::dec << "\n";
    }
    std::cout << "\n";

    // Generate Merkle tree
    std::cout << "========================================\n";
    std::cout << "Generating Merkle Tree\n";
    std::cout << "========================================\n";
    tree_type merkle;
    merkle.allocate_tree();
    uint32_t* d_root;
    uint32_t* root = new uint32_t[merkle.digest_size];
    cudaMalloc(reinterpret_cast<void**>(&d_root), sizeof(uint32_t) * merkle.digest_size);
    generate_tree<MERKLE, HASH, uint32_t>(msg, in_len, merkle, d_root);
    
    // Copy root to host
    cudaMemcpy(root, d_root, sizeof(uint32_t) * merkle.digest_size, cudaMemcpyDeviceToHost);
    
    std::cout << "Merkle tree generated successfully!\n";
    std::cout << "Tree root hash:\n";
    for(uint32_t i = 0; i < merkle.digest_size; i++) {
        std::cout << "  Root[" << i << "]: 0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << root[i] << std::dec << "\n";
    }
    std::cout << "\n"; 

    // Single proof generation and verification (sequential)
    std::cout << "========================================\n";
    std::cout << "Single Proof Generation and Verification\n";
    std::cout << "========================================\n";
    std::cout << "(Sequential: one proof at a time)\n\n";
    
    uint32_t proof_leaf_index = 0;
    std::cout << "Generating proof for leaf at index " << proof_leaf_index << "...\n";
    
    proof_type first_proof;
    first_proof.allocate_proof();
    uint32_t* leaf_to_prove; 

    cudaMalloc(reinterpret_cast<void**>(&leaf_to_prove), proof_type::digest_size * sizeof(uint32_t));
    cudaMemcpy(leaf_to_prove, merkle.nodes, proof_type::digest_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    // Time the sequential proof generation
    auto start_time = std::chrono::high_resolution_clock::now();
    single_proof_kernel<MERKLE, HASH, uint32_t><<<1, 1>>>(first_proof, leaf_to_prove, proof_leaf_index, merkle);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto single_proof_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    auto duration = single_proof_time;
    
    std::cout << "Proof generated successfully!\n";
    std::cout << "  Time: " << duration.count() << " microseconds (" 
              << (duration.count() / 1000.0) << " ms)\n\n";

    std::cout << "Verifying proof for leaf at index " << proof_leaf_index << "...\n";
    bool verified = false;
    bool* d_verified;
    cudaMalloc(reinterpret_cast<void**>(&d_verified), sizeof(bool));
    cudaMemcpy(d_verified, &verified, sizeof(bool), cudaMemcpyHostToDevice);
    
    // Time the sequential proof verification
    start_time = std::chrono::high_resolution_clock::now();
    single_verify_kernel<MERKLE, HASH, uint32_t><<<1, 1>>>(first_proof, leaf_to_prove, proof_leaf_index, d_root, d_verified);
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    auto single_verify_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    duration = single_verify_time;
    
    cudaMemcpy(&verified, d_verified, sizeof(bool), cudaMemcpyDeviceToHost);
    
    if (verified) {
        std::cout << "✓ Proof verification: VALID\n";
        std::cout << "  Leaf " << proof_leaf_index << " is confirmed to be a member of the Merkle tree.\n";
    } else {
        std::cout << "✗ Proof verification: INVALID\n";
        std::cout << "  Leaf " << proof_leaf_index << " verification failed.\n";
    }
    std::cout << "  Verification time: " << duration.count() << " microseconds (" 
              << (duration.count() / 1000.0) << " ms)\n\n";

    cudaFree(leaf_to_prove);
    cudaFree(d_verified);
    first_proof.free_proof();

    // Batch proof generation and verification (parallel)
    std::cout << "========================================\n";
    std::cout << "Batch Proof Generation and Verification\n";
    std::cout << "========================================\n";
    
    constexpr size_t num_proofs = 256;
    std::cout << "(Parallel: " << num_proofs << " proofs generated simultaneously using multiple GPU threads)\n\n";
    std::cout << "Generating proofs for " << num_proofs << " leaves in parallel...\n";
    
    uint32_t* d_leaves_to_prove;
    uint32_t* d_indices_to_prove;
    cudaMalloc(reinterpret_cast<void**>(&d_leaves_to_prove), num_proofs * proof_type::digest_size * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_indices_to_prove), num_proofs * sizeof(uint32_t));
    cudaMemcpy(d_leaves_to_prove, merkle.nodes, num_proofs * proof_type::digest_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    index_set<<<1, num_proofs>>>(d_indices_to_prove, num_proofs);

    // Allocate proofs using cudaMallocManaged
    proof_type* proofs;
    cudaMallocManaged(reinterpret_cast<void**>(&proofs), num_proofs * sizeof(proof_type));
    for (size_t i = 0; i < num_proofs; i++) {
        cudaMallocManaged(reinterpret_cast<void**>(&proofs[i].nodes), N * proof_type::digest_size * sizeof(uint32_t));
        cudaMallocManaged(reinterpret_cast<void**>(&proofs[i].indices), N * sizeof(uint32_t));
    }
    cudaDeviceSynchronize();

    // Time the parallel proof generation
    start_time = std::chrono::high_resolution_clock::now();
    generate_proof<MERKLE, HASH, uint32_t>(d_leaves_to_prove, d_indices_to_prove, num_proofs, proofs, merkle);
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    auto batch_proof_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    duration = batch_proof_time;
    
    std::cout << "Proofs generated successfully!\n";
    std::cout << "  Time: " << duration.count() << " microseconds (" 
              << (duration.count() / 1000.0) << " ms)\n";
    std::cout << "  Average time per proof: " << (duration.count() / static_cast<double>(num_proofs)) 
              << " microseconds\n\n";

    std::cout << "Verifying " << num_proofs << " proofs in parallel...\n";
    cudaMalloc(reinterpret_cast<void**>(&d_verified), num_proofs * sizeof(bool));
    cudaMemset(d_verified, 0, num_proofs * sizeof(bool));
    
    // Time the parallel proof verification
    start_time = std::chrono::high_resolution_clock::now();
    verify_proof<MERKLE, HASH, uint32_t>(d_leaves_to_prove, d_indices_to_prove, num_proofs, proofs, d_root, d_verified);
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    auto batch_verify_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    duration = batch_verify_time;
    
    std::cout << "  Verification time: " << duration.count() << " microseconds (" 
              << (duration.count() / 1000.0) << " ms)\n";
    std::cout << "  Average time per verification: " << (duration.count() / static_cast<double>(num_proofs)) 
              << " microseconds\n\n";
    
    bool verify[num_proofs];
    cudaMemcpy(verify, d_verified, num_proofs * sizeof(bool), cudaMemcpyDeviceToHost);
    
    // Count verified proofs
    size_t verified_count = 0;
    for (size_t i = 0; i < num_proofs; i++) {
        if (verify[i]) verified_count++;
    }
    
    std::cout << "Verification results:\n";
    std::cout << "  ✓ Verified: " << verified_count << " / " << num_proofs << "\n";
    std::cout << "  ✗ Failed: " << (num_proofs - verified_count) << " / " << num_proofs << "\n";
    
    if (verified_count == num_proofs) {
        std::cout << "  All proofs verified successfully!\n";
    }
    std::cout << "\n";
    
    // Performance comparison
    std::cout << "========================================\n";
    std::cout << "Performance Comparison\n";
    std::cout << "========================================\n";
    double sequential_total = single_proof_time.count() + single_verify_time.count();
    double parallel_total = batch_proof_time.count() + batch_verify_time.count();
    double parallel_avg_per_proof = parallel_total / num_proofs;
    double sequential_total_for_batch = sequential_total * num_proofs;
    double speedup = sequential_total / parallel_avg_per_proof;
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Sequential (1 proof at a time):\n";
    std::cout << "  Proof generation:    " << single_proof_time.count() << " μs\n";
    std::cout << "  Proof verification:  " << single_verify_time.count() << " μs\n";
    std::cout << "  Total per proof:     " << sequential_total << " μs\n";
    std::cout << "  Time for " << num_proofs << " proofs: " << (sequential_total_for_batch / 1000.0) << " ms\n\n";
    
    double parallel_proof_avg = batch_proof_time.count() / static_cast<double>(num_proofs);
    double parallel_verify_avg = batch_verify_time.count() / static_cast<double>(num_proofs);
    
    std::cout << "Parallel (" << num_proofs << " proofs simultaneously):\n";
    std::cout << "  Proof generation:    " << batch_proof_time.count() << " μs (avg: " << parallel_proof_avg << " μs/proof)\n";
    std::cout << "  Proof verification:  " << batch_verify_time.count() << " μs (avg: " << parallel_verify_avg << " μs/proof)\n";
    std::cout << "  Total per proof:     " << parallel_avg_per_proof << " μs\n";
    std::cout << "  Time for " << num_proofs << " proofs: " << (parallel_total / 1000.0) << " ms\n\n";
    
    std::cout << "Results:\n";
    std::cout << "  Speedup:             " << speedup << "x faster per proof\n";
    std::cout << "\n";

    // Cleanup
    for (size_t i = 0; i < num_proofs; i++) {
        cudaFree(proofs[i].nodes);
        cudaFree(proofs[i].indices);
    }
    cudaFree(proofs);
    cudaFree(d_leaves_to_prove);
    cudaFree(d_indices_to_prove);
    cudaFree(d_verified);
    delete[] root;

    merkle.free_tree();
    cudaFree(d_root);

    return 0;
}
