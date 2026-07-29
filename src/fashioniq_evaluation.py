from collections import Counter
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


def compute_recall_at_k(
        predicted_features: torch.Tensor,
        target_names: Sequence[str],
        index_features: torch.Tensor,
        index_names: Sequence[str],
        ks: Tuple[int, ...] = (1, 5, 10),
) -> Tuple[float, ...]:
    """Compute percentage recall for labeled FashionIQ-style retrieval queries."""
    if predicted_features.ndim != 2 or index_features.ndim != 2:
        raise ValueError("predicted and index features must both be 2-D tensors")
    if predicted_features.shape[0] != len(target_names):
        raise ValueError(
            "query feature count must match target name count: "
            f"{predicted_features.shape[0]} != {len(target_names)}")
    if index_features.shape[0] != len(index_names):
        raise ValueError(
            "gallery feature count must match index name count: "
            f"{index_features.shape[0]} != {len(index_names)}")
    if predicted_features.shape[1] != index_features.shape[1]:
        raise ValueError("predicted and index feature dimensions must match")
    if not target_names:
        raise ValueError("at least one labeled query is required")
    if any(k <= 0 for k in ks):
        raise ValueError("recall cutoffs must be positive")

    duplicate_names = sorted(
        name for name, count in Counter(index_names).items() if count > 1)
    if duplicate_names:
        raise ValueError(
            "duplicate gallery names are not supported: "
            + ", ".join(duplicate_names))

    gallery_positions = {name: index for index, name in enumerate(index_names)}
    missing_targets = sorted({name for name in target_names if name not in gallery_positions})
    if missing_targets:
        raise ValueError(
            "target names missing from gallery: " + ", ".join(missing_targets))

    predicted_features = F.normalize(predicted_features.float(), dim=-1)
    index_features = F.normalize(index_features.float(), dim=-1)
    ranked_indices = torch.argsort(
        predicted_features @ index_features.T, dim=-1, descending=True)
    target_indices = torch.tensor(
        [gallery_positions[name] for name in target_names],
        device=ranked_indices.device,
    ).unsqueeze(1)
    target_matches = ranked_indices.eq(target_indices)

    return tuple(
        target_matches[:, :min(k, len(index_names))].any(dim=1).float().mean().item() * 100
        for k in ks
    )
