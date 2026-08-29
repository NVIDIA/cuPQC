# cuPQC SDK Applications

Application implementations demonstrating NVIDIA cuPQC SDK capabilities.

## Applications

| Application | Description |
|-------------|-------------|
| [Merkle Tree Membership Proof (Poseidon2 BB)](merkle_proof_poseidon2_bb/) | Complete membership proof system with tree generation, proof creation, and verification using Poseidon2 (BabyBear field) |

## Building

Each application includes a Makefile:

```bash
cd applications/<app-name>
make
./<executable-name>
```

**Note:** Each application has its own executable name. See individual application READMEs for the exact executable name.

See individual application READMEs for detailed instructions and prerequisites.
