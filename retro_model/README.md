---
language:
- en
license: mit
tags:
- chemistry
- retrosynthesis
- mechanism-aware
- latent-thinking
- rdkit
pipeline_tag: feature-extraction
---

# Dynamic Mechanism-Aware Chemical Retrosynthesis Model

This model implements high-dimensional latent chemical thinking for organic retrosynthesis.
Rather than attempting character-token seq2seq generation, it operates in a continuous chemical manifold (R^256) where multi-step reaction reasoning is mathematically forced through genuine reaction intermediates before decoding starting materials.

## Key Architectural Highlights
- Dynamic Pathway Depth: Dynamically unrolls for variable intermediate steps (1 <= M_i <= 6) with learned dynamic halting (p_halt).
- Zero Truncation: Dense 1040-dim representation captures any molecule of arbitrary size without token length caps.
- Hypersphere Manifold: L2-normalized representations trained via contrastive InfoNCE (tau=0.07).

## Files Included
- `model.safetensors` - Safe, fast tensor weights for the projector and dynamic retro model.
- `pytorch_model.bin` - Standard PyTorch checkpoint weights.
- `config.json` - Hugging Face model architecture specification.
- `generation_config.json` - Thought unrolling and candidate retrieval hyperparameters.
- `featurizer_config.json` - Circular Morgan fingerprint (1024-bit, radius 2) + 16 physicochemical descriptors.
- `vocab.json` & `tokenizer_config.json` - Chemistry symbol vocabulary.
- `special_tokens_map.json` - Special token mappings.

## Usage Example
```python
import torch
import safetensors.torch
from rdkit import Chem

# Load weights
weights = safetensors.torch.load_file("model.safetensors")
print("Model loaded successfully, total parameters:", len(weights))
```
