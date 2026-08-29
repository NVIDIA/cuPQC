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

#include <cstdlib>
#include <vector>
#include <iostream>

#include <ntt.hpp>

using namespace cupqc;

/*
 * NTT type aliases.
 * Polynomial degree N=1024 over the KoalaBear prime field (p = 2^31 - 2^24 + 1).
 * The operator composition selects algorithm, direction, element precision, and
 * polynomial size at compile time.
 */
constexpr uint32_t NTT_N = 1024;

using ForwardNTT = decltype(Algorithm<algorithm::NTT>()
                            + Direction<nttDirection::FORWARD>()
                            + Precision<uint32_t>()
                            + Size<NTT_N>()
                            + Block()
                            + BlockDim<128>());

using InverseNTT = decltype(Algorithm<algorithm::NTT>()
                            + Direction<nttDirection::INVERSE>()
                            + Precision<uint32_t>()
                            + Size<NTT_N>()
                            + Block()
                            + BlockDim<128>());

// Host-side modular exponentiation: base^exp mod m.
// Used to compute g^{p-2} mod p = g^{-1} mod p (Fermat's little theorem).
static uint32_t host_modpow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) { result = result * base % mod; }
        base = base * base % mod;
        exp >>= 1;
    }
    return static_cast<uint32_t>(result);
}

/*
 * Kernel: generate twiddle factors for one NTT direction.
 * Runs single-threaded (<<<1,1>>>) — called once per prime/generator pair.
 * Forward twiddles use g; inverse twiddles use g^{-1}.
 */
template<class NTT>
__global__ void make_twiddles_kernel(uint32_t* twiddles, const uint32_t p, const uint32_t g) {
    NTT().make_twiddles(twiddles, p, g);
}
template<class NTT>
__global__ void transform_twiddles_to_mont_kernel(uint32_t* twiddles, const uint32_t p) {
    NTT().transform_twiddles_to_mont(twiddles, p);
}

/*
 * Kernel: run a forward NTT on one polynomial per CUDA block.
 * blockIdx.x selects the polynomial in the batch.
 */
__global__ void forward_ntt_kernel(uint32_t* polys, const uint32_t* twiddles, const nttConst<uint32_t> ntt_const) {
    uint32_t* poly = polys + blockIdx.x * ForwardNTT::Size;
    extern __shared__ uint32_t sdata[];
    ForwardNTT().load_to_mont(sdata, poly, ntt_const);
    __syncthreads();
    ForwardNTT().execute(sdata, twiddles, ntt_const.p);
    __syncthreads();
    ForwardNTT().store_from_mont(sdata, poly, ntt_const);
}

/*
 * Kernel: run an inverse NTT on one polynomial per CUDA block.
 * N_inv = N^{-1} mod p is the scaling factor that normalises the INTT output.
 */
__global__ void inverse_ntt_kernel(uint32_t* polys, const uint32_t* inv_twiddles,
                                    const nttConst<uint32_t> ntt_const, const uint32_t N_inv) {
    uint32_t* poly = polys + blockIdx.x * InverseNTT::Size;
    extern __shared__ uint32_t sdata[];
    InverseNTT().load_to_mont(sdata, poly, ntt_const);
    __syncthreads();
    InverseNTT().execute(sdata, inv_twiddles, ntt_const.p, N_inv);
    __syncthreads();
    InverseNTT().store_from_mont(sdata, poly, ntt_const);
}

/*
 * Runs a batched NTT round-trip and verifies INTT(NTT(f)) == f.
 *
 * Twiddle tables are generated once on the device and reused across the
 * entire batch.  Each CUDA block processes one polynomial independently.
 *
 * Parameters:
 *   batch   - number of polynomials to transform
 *   p       - prime modulus
 *   g       - primitive N-th root of unity mod p  (forward generator)
 *   g_inv   - g^{-1} mod p                        (inverse generator)
 *   N_inv   - N^{-1} mod p                        (INTT normalisation factor)
 */
bool ntt_round_trip(const unsigned int batch,
                     const uint32_t p,
                     const uint32_t g,
                     const uint32_t g_inv,
                     const uint32_t N_inv)
{
    constexpr uint32_t N    = ForwardNTT::Size;
    constexpr size_t   smem = ntt_shared_workspace_size<N, uint32_t>();

    /*
     * Initialise batch polynomials with values 0, 1, 2, ... (mod p).
     */
    std::vector<uint32_t> h_data(N * batch);
    for (unsigned int b = 0; b < batch; b++) {
        for (uint32_t i = 0; i < N; i++) {
            h_data[b * N + i] = (b * N + i) % p;
        }
    }
    const std::vector<uint32_t> reference = h_data;

    /*
     * Allocate device memory for polynomials and twiddle tables.
     */
    uint32_t* d_data        = nullptr;
    uint32_t* d_twiddles     = nullptr;
    uint32_t* d_inv_twiddles = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_data),        N * batch * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_twiddles),     N * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_inv_twiddles), N * sizeof(uint32_t));

    cudaMemcpy(d_data, h_data.data(), N * batch * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /*
     * Generate twiddle tables once — shared across all polynomials in the batch.
     */
    make_twiddles_kernel<ForwardNTT><<<1, 1>>>(d_twiddles,     p, g);
    make_twiddles_kernel<InverseNTT><<<1, 1>>>(d_inv_twiddles, p, g_inv);
    transform_twiddles_to_mont_kernel<ForwardNTT><<<1, ForwardNTT::BlockDim>>>(d_twiddles, p);
    transform_twiddles_to_mont_kernel<InverseNTT><<<1, InverseNTT::BlockDim>>>(d_inv_twiddles, p);

    const nttConst<uint32_t> ntt_const(p);
    /*
     * Forward NTT: transform each polynomial into its NTT representation.
     * One CUDA block per polynomial; shared memory holds the working buffer.
     */
    forward_ntt_kernel<<<batch, ForwardNTT::BlockDim, smem>>>(d_data, d_twiddles, ntt_const);

    /*
     * Inverse NTT: recover the original polynomials.
     */
    inverse_ntt_kernel<<<batch, InverseNTT::BlockDim, smem>>>(d_data, d_inv_twiddles, ntt_const, N_inv);

    /*
     * Transfer results back to host.
     */
    cudaMemcpy(h_data.data(), d_data, N * batch * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_twiddles);
    cudaFree(d_inv_twiddles);

    /*
     * Verify: INTT(NTT(f)) == f for every coefficient of every polynomial.
     */
    unsigned int errors = 0;
    for (size_t i = 0; i < reference.size(); i++) {
        if (h_data[i] != reference[i]) {
            errors++;
        }
    }
    if (errors == 0) {
        std::cout << "Round-trip OK: " << batch << " polynomial(s) of degree " << N << " verified." << std::endl;
        return true;
    }
    std::cout << "Round-trip FAILED: " << errors << " coefficient(s) did not match." << std::endl;
    return false;
}

/*
 * In this example we perform a batched NTT round-trip over the KoalaBear
 * prime field.  The KoalaBear prime p = 2^31 - 2^24 + 1 supports NTT sizes
 * up to 2^24.  Here we use N = 1024 = 2^10.
 *
 * The high-level flow for each polynomial in the batch:
 *   1. Forward NTT  — convert coefficient representation to evaluation form.
 *   2. Inverse NTT  — convert back to coefficient representation.
 *   3. Verify       — confirm the output equals the original input.
 *
 * In a real application the NTT and INTT steps would bracket a pointwise
 * multiplication (i.e. polynomial multiplication via NTT).
 */
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "================================================================\n";
    std::cout << "Number Theoretic Transform (NTT) Example\n";
    std::cout << "================================================================\n\n";

    std::cout << "This example demonstrates a batched NTT round-trip using cuPQC SDK.\n";
    std::cout << "The NTT is the modular-arithmetic analogue of the FFT, and is the\n";
    std::cout << "standard way to turn polynomial multiplication into cheap pointwise\n";
    std::cout << "multiplication for lattice cryptography, FHE, and ZK proof systems.\n\n";
    std::cout << "Configuration: N=" << NTT_N << ", KoalaBear field, block execution\n";
    std::cout << "Layout:        one polynomial per CUDA block, shared-memory workspace\n\n";

    /*
     * KoalaBear prime: p = 2^31 - 2^24 + 1 = 2130706433.
     * KoalaBear_primitive_root_10 is a primitive 2^10-th root of unity mod p.
     */
    constexpr uint32_t p = cupqc::KoalaBear;
    constexpr uint32_t g = cupqc::KoalaBear_primitive_root_10;

    // g^{-1} mod p = g^{p-2} mod p  (Fermat's little theorem)
    const uint32_t g_inv = host_modpow(g, p - 2, p);
    // N^{-1} mod p, applied by the inverse NTT to normalise the output
    const uint32_t N_inv = n_inv<NTT_N>(p);

    const unsigned int batch = 8;
    const bool ok = ntt_round_trip(batch, p, g, g_inv, N_inv);

    std::cout << "\n" << (ok ? "Example completed successfully.\n"
                             : "Example failed.\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
