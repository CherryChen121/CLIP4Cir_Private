import pytest
import torch

from fashioniq_evaluation import compute_recall_at_k


def test_compute_recall_at_k_uses_target_ranks():
    recalls = compute_recall_at_k(
        predicted_features=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        target_names=["a", "b"],
        index_features=torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
        index_names=["a", "b", "c"],
        ks=(1, 2, 3),
    )

    assert recalls == pytest.approx((50.0, 100.0, 100.0))


def test_compute_recall_at_k_rejects_missing_target():
    with pytest.raises(ValueError, match="missing"):
        compute_recall_at_k(
            predicted_features=torch.tensor([[1.0, 0.0]]),
            target_names=["missing"],
            index_features=torch.tensor([[1.0, 0.0]]),
            index_names=["present"],
        )


def test_compute_recall_at_k_rejects_duplicate_gallery_names():
    with pytest.raises(ValueError, match="duplicate.*a"):
        compute_recall_at_k(
            predicted_features=torch.tensor([[1.0, 0.0]]),
            target_names=["a"],
            index_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            index_names=["a", "a"],
        )


def test_compute_recall_at_k_rejects_mismatched_query_count():
    with pytest.raises(ValueError, match="query"):
        compute_recall_at_k(
            predicted_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            target_names=["a"],
            index_features=torch.tensor([[1.0, 0.0]]),
            index_names=["a"],
        )
