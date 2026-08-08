"""Ground-truth constants for the Kimi-K3 checkpoint and target fleet.

Every value here was verified against the real checkpoint
(moonshotai/Kimi-K3 @ 9f62e4e9, model.safetensors.index.json + shard headers)
on 2026-07-27. Inventory recomputes byte totals from shard headers at runtime;
these constants exist so the rest of the code never re-derives naming or
geometry.
"""

from __future__ import annotations

MODEL_ID = "moonshotai/Kimi-K3"
REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"

# ---------------------------------------------------------------------------
# Architecture geometry (config.json, verified against shard headers)
# ---------------------------------------------------------------------------
NUM_LAYERS = 93
MOE_LAYERS = tuple(range(1, 93))  # layer 0 is dense (first_k_dense_replace=1)
NUM_MOE_LAYERS = len(MOE_LAYERS)  # 92
NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 7168
LATENT = 3584  # routed_expert_hidden_size (LatentMoE)
EXPERT_INTER = 3072  # moe_intermediate_size

# Expert matrices, logical [out, in] orientation as stored:
#   w1 (gate): [3072, 3584]   w3 (up): [3072, 3584]   w2 (down): [3584, 3072]
EXPERT_MATRICES = ("w1", "w3", "w2")
EXPERT_SHAPES = {
    "w1": (EXPERT_INTER, LATENT),
    "w3": (EXPERT_INTER, LATENT),
    "w2": (LATENT, EXPERT_INTER),
}

# ---------------------------------------------------------------------------
# Tensor naming (index.json ground truth)
# ---------------------------------------------------------------------------
LM_PREFIX = "language_model.model."
INDEX_FILE = "model.safetensors.index.json"


def expert_tensor(layer: int, expert: int, matrix: str, part: str) -> str:
    """part is 'weight_packed' or 'weight_scale'."""
    return (
        f"{LM_PREFIX}layers.{layer}.block_sparse_moe.experts.{expert}"
        f".{matrix}.{part}"
    )


def router_bias_tensor(layer: int) -> str:
    return f"{LM_PREFIX}layers.{layer}.block_sparse_moe.gate.e_score_correction_bias"


def router_weight_tensor(layer: int) -> str:
    return f"{LM_PREFIX}layers.{layer}.block_sparse_moe.gate.weight"


def latent_up_proj_tensor(layer: int) -> str:
    return f"{LM_PREFIX}layers.{layer}.block_sparse_moe.routed_expert_up_proj.weight"


# ---------------------------------------------------------------------------
# Source quantization format: compressed-tensors "mxfp4-pack-quantized"
#   weight_packed: uint8 [out, in/2], two E2M1 codes per byte, low nibble first
#   weight_scale:  uint8 [out, in/32], E8M0 (value = 2^(byte - 127)), K-blocks
# ---------------------------------------------------------------------------
MXFP4_BLOCK = 32
E8M0_BIAS = 127
# E2M1 code -> value; codes 8..15 are the negated mirror (8 == -0).
E2M1_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
MXFP4_BITS_EFFECTIVE = 4.25  # 4b code + 8b scale / 32

# NF2 contingency: 4-level sign-magnitude codebook (refittable).
NF2_CODEBOOK = (-1.0, -0.3563, 0.3563, 1.0)
NF2_BITS_EFFECTIVE = 2.25

EXL3_BITS_EFFECTIVE = 3.0  # trellis, PR sparkinfer#49 arm (3/4/5/6 bpw only)

# ---------------------------------------------------------------------------
# Fleet ledger defaults (12x RTX PRO 6000 Blackwell, TP=12, single node).
# All byte values binary (GiB). budget.py prefers live inventory numbers;
# these are the documented defaults and CLI fallbacks.
# ---------------------------------------------------------------------------
GIB = 2**30
FLEET_GPUS = 12
GPU_USABLE_GIB = 95.59  # 97887 MiB per nvidia-smi
FLEET_GIB = FLEET_GPUS * GPU_USABLE_GIB  # 1147.1 GiB

# Non-expert serving plan (dtype decisions from the design discussion):
#   attn/KDA + shared experts + dense L0 -> MXFP8 (8.25 bpw)
#   latent projections (ship BF16) -> MXFP8
#   router weight + bias, norms, lm_head, embeddings, vision -> BF16
# Router is replicated on every rank under TP.
DEFAULT_RUNTIME_RESERVE_GIB = 60.0
DEFAULT_CONTEXT_RESERVE_GIB = 12.0  # ~2-3 concurrent 256K streams with DCP4
DEFAULT_TARGET_BPW = 3.20

TOTAL_EXPERT_PARAMS = NUM_MOE_LAYERS * NUM_EXPERTS * (
    2 * EXPERT_INTER * LATENT + LATENT * EXPERT_INTER
)  # 2.7227e12
