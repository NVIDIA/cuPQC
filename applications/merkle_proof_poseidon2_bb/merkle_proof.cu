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

/**
 * Merkle Tree Membership Proof - Zero-Knowledge Proof Application
 * 
 * Demonstrates zero-knowledge membership proofs using Merkle trees.
 * This application generates a large Merkle tree and demonstrates how to
 * generate and verify membership proofs without revealing the specific element.
 */

#include <vector>
#include <iomanip>
#include <iostream>
#include <random>

#include <hash.hpp>

using namespace cupqc;

/* 
 * Merkle Tree Membership Proof Application
 * 
 * This application demonstrates:
 * 1. Generating a large Merkle tree from a set of data elements
 * 2. Generating a membership proof for a specific element
 * 3. Verifying the proof against the stored root hash
 * 4. Demonstrating zero-knowledge property (doesn't reveal other elements)
 */

using HASH   = decltype(POSEIDON2_BB_8_16() + Thread());
using MERKLE = decltype(MERKLE_FIELD_2097152() + BlockDim<256>());
using SUB_MERKLE = decltype(MERKLE_FIELD_2048() + BlockDim<256>());
using FINAL_MERKLE = decltype(MERKLE_FIELD_1024() + BlockDim<256>());

/**
 * Kernel to create leaves of the Merkle tree
 */
__global__ void create_leaves_kernel(tree<MERKLE::Size, HASH, uint32_t> merkle, const uint32_t* msg, size_t inbuf_len)
{
    HASH hash{};
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < MERKLE::Size) {
        MERKLE().create_leaf(merkle.nodes + i * merkle.digest_size, msg + i * inbuf_len, hash, inbuf_len);
    }
}

/**
 * Kernel to generate sub-trees in parallel
 */
__global__ void generate_sub_tree_kernel(tree<MERKLE::Size, HASH, uint32_t> merkle)
{
    HASH hash{};
    SUB_MERKLE().generate_sub_tree(hash, merkle, blockIdx.x);
}

/**
 * Kernel to generate the final tree from sub-trees
 */
__global__ void generate_final_tree_kernel(tree<MERKLE::Size, HASH, uint32_t> merkle, uint32_t* root)
{
    HASH hash;
    tree<FINAL_MERKLE::Size, HASH, uint32_t> merkle_final;
    const size_t left_overs = merkle.size - merkle_final.size;
    merkle_final.nodes = merkle.nodes + left_overs * merkle.digest_size;
    
    __syncthreads();
    FINAL_MERKLE().generate_tree(hash, merkle_final);

    if(threadIdx.x == 0) {
        for(uint32_t i = 0; i < merkle.digest_size; i++) {
            root[i] = merkle.root()[i];
        }
    }
}

/**
 * Generate Merkle tree from messages
 * msg: Vector of messages (each element is a leaf)
 * inbuf_len: Length of each message
 * merkle: Merkle tree structure
 * d_root: Device pointer to store root hash
 */
void generate_tree(const std::vector<uint32_t>& msg, size_t inbuf_len, tree<MERKLE::Size, HASH, uint32_t> merkle, uint32_t* d_root)
{
    uint32_t* d_msg;
    cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size() * sizeof(uint32_t));
    cudaMemcpy(d_msg, msg.data(), msg.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    create_leaves_kernel<<<MERKLE::Size / MERKLE::BlockDim.x, MERKLE::BlockDim>>>(merkle, d_msg, inbuf_len);
    generate_sub_tree_kernel<<<FINAL_MERKLE::Size, MERKLE::BlockDim>>>(merkle);
    generate_final_tree_kernel<<<1, MERKLE::BlockDim>>>(merkle, d_root);
    
    cudaFree(d_msg);
}

/**
 * Kernel to generate a proof for a specific leaf
 * proof_obj: Proof object to populate
 * leaf: Leaf data to generate proof for
 * leaf_index: Index of the leaf
 * merkle: Merkle tree structure
 */
__global__ void generate_proof_kernel(proof<MERKLE::Size, HASH, uint32_t> proof_obj, 
                                      const uint32_t* leaf, 
                                      const uint32_t leaf_index, 
                                      const tree<MERKLE::Size, HASH, uint32_t> merkle)
{
    MERKLE().generate_proof(proof_obj, leaf, leaf_index, merkle);
}

/**
 * Kernel to verify a proof against the root
 * proof_obj: Proof object to verify
 * leaf: Leaf data being proven
 * leaf_index: Index of the leaf
 * root: Root hash to verify against
 * verified: Output boolean indicating if proof is valid
 */
__global__ void verify_proof_kernel(const proof<MERKLE::Size, HASH, uint32_t> proof_obj,
                                     const uint32_t* leaf,
                                     const uint32_t leaf_index,
                                     const uint32_t* root,
                                     bool* verified)
{
    HASH hash{};
    *verified = MERKLE().verify_proof(proof_obj, leaf, leaf_index, root, hash);
}

/**
 * Generate a membership proof for a specific leaf
 * leaf_index: Index of the leaf to generate proof for
 * merkle: Merkle tree structure
 * proof_obj: Proof object to populate
 */
void generate_proof_for_leaf(uint32_t leaf_index,
                             const tree<MERKLE::Size, HASH, uint32_t>& merkle,
                             proof<MERKLE::Size, HASH, uint32_t>& proof_obj)
{
    // Get the leaf from the tree
    uint32_t* d_leaf;
    cudaMalloc(reinterpret_cast<void**>(&d_leaf), merkle.digest_size * sizeof(uint32_t));
    cudaMemcpy(d_leaf, merkle.nodes + leaf_index * merkle.digest_size, 
               merkle.digest_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    // Generate the proof
    generate_proof_kernel<<<1, 1>>>(proof_obj, d_leaf, leaf_index, merkle);
    cudaDeviceSynchronize();
    
    cudaFree(d_leaf);
}


/**
 * Verify a proof against the root hash
 * proof_obj: Proof to verify
 * leaf_index: Index of the leaf being proven
 * merkle: Merkle tree structure
 * root: Root hash to verify against
 * Returns: true if proof is valid, false otherwise
 */
bool verify_proof(const proof<MERKLE::Size, HASH, uint32_t>& proof_obj,
                  uint32_t leaf_index,
                  const tree<MERKLE::Size, HASH, uint32_t>& merkle,
                  const uint32_t* root)
{
    // Get the leaf from the tree
    uint32_t* d_leaf;
    bool* d_verified;
    bool verified = false;
    
    cudaMalloc(reinterpret_cast<void**>(&d_leaf), merkle.digest_size * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_verified), sizeof(bool));
    
    cudaMemcpy(d_leaf, merkle.nodes + leaf_index * merkle.digest_size, 
               merkle.digest_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    // Verify the proof
    verify_proof_kernel<<<1, 1>>>(proof_obj, d_leaf, leaf_index, root, d_verified);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&verified, d_verified, sizeof(bool), cudaMemcpyDeviceToHost);
    
    cudaFree(d_leaf);
    cudaFree(d_verified);
    
    return verified;
}

/**
 * Main application entry point
 */
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "================================================================\n";
    std::cout << "Merkle Tree Membership Proof - Zero-Knowledge Proof Application\n";
    std::cout << "================================================================\n\n";

    constexpr auto N = MERKLE::Size;
    constexpr size_t in_len = 64;
    
    std::cout << "Generating Merkle tree with " << N << " elements...\n";
    std::cout << "Each element has length " << in_len << " (BabyBear field elements)\n";
    
    // Allocate tree to get digest_size
    tree<MERKLE::Size, HASH, uint32_t> merkle;
    merkle.allocate_tree();
    
    std::cout << "Raw leaf data size: " << (in_len * sizeof(uint32_t)) << " bytes (" << in_len << " × " << sizeof(uint32_t) << " bytes)\n";
    std::cout << "Leaf hash size (stored in tree): " << (merkle.digest_size * sizeof(uint32_t)) << " bytes (" << merkle.digest_size << " × " << sizeof(uint32_t) << " bytes)\n\n";

    // Generate sample data for the Merkle tree
    // In a real application, this would be actual data elements
    std::vector<uint32_t> msg(in_len * N, 0);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < in_len; j++) {
            msg[i * in_len + j] = (i * j) % cupqc_common::BabyBearPrime;
        }
    }

    // Merkle tree already allocated above
    
    uint32_t* d_root = nullptr;
    uint32_t* root = new uint32_t[merkle.digest_size];
    cudaMalloc(reinterpret_cast<void**>(&d_root), merkle.digest_size * sizeof(uint32_t));
    
    // Generate the tree
    generate_tree(msg, in_len, merkle, d_root);
    
    // Copy root to host
    cudaMemcpy(root, d_root, merkle.digest_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    std::cout << "Merkle tree generated successfully!\n";
    std::cout << "Root hash:\n";
    for(uint32_t i = 0; i < merkle.digest_size; i++) {
        std::cout << "  Root[" << i << "]: 0x" << std::hex << std::setw(8) << std::setfill('0') << root[i] << std::dec << "\n";
    }
    std::cout << "\n";

    // Generate and verify a proof for a specific leaf
    std::cout << "========================================\n";
    std::cout << "Generating Membership Proof\n";
    std::cout << "========================================\n";
    
    // Select a random leaf index
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, N - 1);
    uint32_t proof_leaf_index = dis(gen);
    
    proof<MERKLE::Size, HASH, uint32_t> proof_obj;
    proof_obj.allocate_proof();
    
    std::cout << "Selected random leaf index: " << proof_leaf_index << " (out of " << N << " leaves)\n";
    std::cout << "Leaf input size: " << (in_len * sizeof(uint32_t)) << " bytes (" << in_len << " × " << sizeof(uint32_t) << " bytes)\n\n";
    
    // Print the leaf data being proven
    std::cout << "Leaf data (first 8 values):\n";
    for (size_t j = 0; j < std::min(in_len, size_t(8)); j++) {
        uint32_t leaf_value = msg[proof_leaf_index * in_len + j];
        std::cout << "  Leaf[" << j << "]: 0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << leaf_value << std::dec << " (" << leaf_value << ")\n";
    }
    if (in_len > 8) {
        std::cout << "  ... (showing first 8 of " << in_len << " values)\n";
    }
    std::cout << "\n";
    
    std::cout << "Generating proof for leaf at index " << proof_leaf_index << "...\n";
    generate_proof_for_leaf(proof_leaf_index, merkle, proof_obj);
    std::cout << "Proof generated successfully!\n\n";
    
    // Verify the proof
    std::cout << "========================================\n";
    std::cout << "Verifying Membership Proof\n";
    std::cout << "========================================\n";
    
    std::cout << "Verifying proof for leaf at index " << proof_leaf_index << "...\n";
    std::cout << "  Leaf data hash (digest):\n";
    
    // Get and print the leaf hash from the tree
    uint32_t* h_leaf = new uint32_t[merkle.digest_size];
    cudaMemcpy(h_leaf, merkle.nodes + proof_leaf_index * merkle.digest_size, 
               merkle.digest_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    for(uint32_t i = 0; i < merkle.digest_size; i++) {
        std::cout << "    LeafHash[" << i << "]: 0x" << std::hex << std::setw(8) 
                  << std::setfill('0') << h_leaf[i] << std::dec << "\n";
    }
    std::cout << "\n";
    
    // Show root comparison
    std::cout << "Root Hash Comparison:\n";
    std::cout << "  Stored Root (from tree):\n";
    for(uint32_t i = 0; i < merkle.digest_size; i++) {
        std::cout << "    Root[" << i << "]: 0x" << std::hex << std::setw(8) 
                  << std::setfill('0') << root[i] << std::dec << "\n";
    }
    std::cout << "  Computed Root (from proof):\n";
    std::cout << "    [Computed internally during verification - matches stored root if valid]\n\n";
    
    bool is_valid = verify_proof(proof_obj, proof_leaf_index, merkle, d_root);
    
    if (is_valid) {
        std::cout << "✓ Proof verification: VALID\n";
        std::cout << "  ✓ Stored root and computed root from proof MATCH\n";
        std::cout << "  The leaf at index " << proof_leaf_index << " is confirmed to be a member of the Merkle tree.\n";
        std::cout << "  The proof correctly demonstrates membership without revealing other leaves.\n";
    } else {
        std::cout << "✗ Proof verification: INVALID\n";
        std::cout << "  ✗ Stored root and computed root from proof DO NOT MATCH\n";
        std::cout << "  The proof does not match the root hash.\n";
    }
    std::cout << "\n";
    
    delete[] h_leaf;
    
    // Demonstrate tampering detection
    std::cout << "========================================\n";
    std::cout << "Tampering Detection Demo\n";
    std::cout << "========================================\n";
    
    // Tamper a different leaf (not the one we proved)
    uint32_t tampered_leaf_index = (proof_leaf_index + 1) % N; // Tamper a different leaf
    std::cout << "Tampering leaf at index " << tampered_leaf_index << " (different from proven leaf " << proof_leaf_index << ")\n";
    
    // Create a tampered version of the message data
    std::vector<uint32_t> tampered_msg = msg;
    // Tamper the first few values of the tampered leaf
    for(size_t j = 0; j < std::min(in_len, size_t(4)); j++) {
        tampered_msg[tampered_leaf_index * in_len + j] = (tampered_msg[tampered_leaf_index * in_len + j] + 1000) % cupqc_common::BabyBearPrime;
    }
    
    std::cout << "  Original leaf data vs Tampered leaf data (first 4 values):\n";
    for(size_t j = 0; j < std::min(in_len, size_t(4)); j++) {
        uint32_t orig_val = msg[tampered_leaf_index * in_len + j];
        uint32_t tampered_val = tampered_msg[tampered_leaf_index * in_len + j];
        std::cout << "    Leaf[" << j << "]: 0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << orig_val << std::dec << " -> 0x" << std::hex << std::setw(8) 
                  << std::setfill('0') << tampered_val << std::dec << "\n";
    }
    std::cout << "\n";
    
    // Generate a new tree with the tampered leaf
    tree<MERKLE::Size, HASH, uint32_t> tampered_merkle;
    tampered_merkle.allocate_tree();
    
    uint32_t* d_tampered_root;
    uint32_t* tampered_root = new uint32_t[merkle.digest_size];
    cudaMalloc(reinterpret_cast<void**>(&d_tampered_root), merkle.digest_size * sizeof(uint32_t));
    
    // Generate tree with tampered data
    generate_tree(tampered_msg, in_len, tampered_merkle, d_tampered_root);
    
    // Copy tampered root to host
    cudaMemcpy(tampered_root, d_tampered_root, merkle.digest_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    std::cout << "Root Hash Comparison: Original vs Tampered (after leaf tampering)\n";
    std::cout << "  Original Root          Tampered Root\n";
    std::cout << "  " << std::string(50, '-') << "\n";
    
    bool roots_differ = false;
    for(uint32_t i = 0; i < merkle.digest_size; i++) {
        bool is_different = (root[i] != tampered_root[i]);
        if(is_different) roots_differ = true;
        std::cout << "  Root[" << i << "]: 0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << root[i] << std::dec;
        std::cout << "    0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << tampered_root[i] << std::dec;
        if(is_different) {
            std::cout << "  ✗";
        } else {
            std::cout << "  ✓";
        }
        std::cout << "\n";
    }
    std::cout << "  " << std::string(50, '-') << "\n";
    std::cout << "\n";
    
    if(roots_differ) {
        std::cout << "✓ Root hash changed after tampering leaf (as expected)\n";
        std::cout << "  The tampered leaf caused the entire root hash to change.\n\n";
    }
    
    std::cout << "Verifying original proof against tampered root...\n";
    std::cout << "  (Proof was generated for original tree, verifying against tampered tree root)\n";
    bool tampered_valid = verify_proof(proof_obj, proof_leaf_index, merkle, d_tampered_root);
    
    if (tampered_valid) {
        std::cout << "✗ Proof verification: VALID (unexpected!)\n";
    } else {
        std::cout << "✓ Proof verification: INVALID (expected)\n";
        std::cout << "  ✗ Tampered root and computed root from proof DO NOT MATCH\n";
        std::cout << "  ✓ Tampering detected! The proof correctly rejects the tampered root.\n";
        std::cout << "  The computed root from proof matches the original root, not the tampered one.\n";
        std::cout << "  This proves that even tampering a different leaf invalidates all proofs.\n";
    }
    std::cout << "\n";
    
    // Cleanup tampered tree
    tampered_merkle.free_tree();
    
    // Cleanup proof
    proof_obj.free_proof();
    cudaFree(d_tampered_root);
    delete[] tampered_root;

    // Cleanup
    merkle.free_tree();
    cudaFree(d_root);
    delete[] root;
    
    std::cout << "Application completed successfully.\n";
    return 0;
}
