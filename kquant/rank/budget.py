"""Fleet ledger -> routed-expert byte budget.

Prefers live inventory params; falls back to constants. All arithmetic in
bytes; GiB only at the presentation edge.
"""

from __future__ import annotations

from dataclasses import dataclass

from kquant import constants as C

GIB = C.GIB

# serving-plan bytes-per-param for non-expert categories
_MXFP8 = 8.25 / 8
_BF16 = 2.0
DEFAULT_SERVING_PLAN = {
    "attention": _MXFP8,
    "shared_experts": _MXFP8,
    "dense_mlp": _BF16,
    "latent_proj": _BF16,
    "router": _BF16,  # sharded over TP (per-rank logits + all-gather)
    "attn_residual": _BF16,
    "embeddings": _BF16,
    "lm_head": _BF16,  # verified UNTIED from embeddings (cos ~0.006)
    "vision": _BF16,
    "norms_other": _BF16,
    "uncategorized": _BF16,
}
# Router is TP-sharded by decision (2026-07-27): each rank computes its 896/12
# logit slice and all-gathers ~1.8 KB/token. No replicated categories remain.
_REPLICATED: set[str] = set()


@dataclass
class Budget:
    fleet_bytes: int
    non_expert_bytes: int
    runtime_reserve_bytes: int
    context_reserve_bytes: int
    expert_budget_bytes: int
    expert_bpw: float
    detail: dict

    def summary(self) -> str:
        d = self.detail
        lines = [
            f"fleet                {self.fleet_bytes / GIB:9.1f} GiB",
            f"- non-expert weights {self.non_expert_bytes / GIB:9.1f} GiB",
        ]
        for cat, b in sorted(d["non_expert"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {cat:16s} {b / GIB:9.2f} GiB")
        lines += [
            f"- runtime reserve    {self.runtime_reserve_bytes / GIB:9.1f} GiB",
            f"- context reserve    {self.context_reserve_bytes / GIB:9.1f} GiB",
            f"= expert budget      {self.expert_budget_bytes / GIB:9.1f} GiB"
            f"  ({self.expert_bpw:.3f} effective bpw)",
        ]
        return "\n".join(lines)


def bpw_to_bytes(bpw: float) -> int:
    return int(C.TOTAL_EXPERT_PARAMS * bpw / 8)


def bytes_to_bpw(nbytes: float) -> float:
    return nbytes * 8 / C.TOTAL_EXPERT_PARAMS


def compute_budget(
    inventory: dict | None = None,
    fleet_gib: float = C.FLEET_GIB,
    runtime_reserve_gib: float = C.DEFAULT_RUNTIME_RESERVE_GIB,
    context_reserve_gib: float = C.DEFAULT_CONTEXT_RESERVE_GIB,
    serving_plan: dict[str, float] | None = None,
) -> Budget:
    plan = dict(DEFAULT_SERVING_PLAN)
    if serving_plan:
        plan.update(serving_plan)

    non_expert = {}
    if inventory:
        for cat, row in inventory["categories"].items():
            if cat == "routed_experts":
                continue
            bpp = plan.get(cat, _BF16)
            b = row["params"] * bpp
            if cat in _REPLICATED:
                b *= C.FLEET_GPUS
            non_expert[cat] = int(b)
    else:
        raise ValueError("inventory required (run `kquant inventory` first)")

    fleet = int(fleet_gib * GIB)
    ne = sum(non_expert.values())
    rr = int(runtime_reserve_gib * GIB)
    cr = int(context_reserve_gib * GIB)
    expert = fleet - ne - rr - cr
    return Budget(
        fleet_bytes=fleet,
        non_expert_bytes=ne,
        runtime_reserve_bytes=rr,
        context_reserve_bytes=cr,
        expert_budget_bytes=expert,
        expert_bpw=bytes_to_bpw(expert),
        detail={"non_expert": non_expert, "serving_plan": plan},
    )
