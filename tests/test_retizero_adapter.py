from types import SimpleNamespace

import torch
from torch import nn

from retizero_adapter import RetiZeroAdapter


class TinyLoRAVisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.linear_a_q = nn.Linear(2, 1, bias=False)
        self.linear_b_q = nn.Linear(1, 2, bias=False)
        self.linear_a_v = nn.Linear(2, 1, bias=False)
        self.linear_b_v = nn.Linear(1, 2, bias=False)
        self.w_As = [self.linear_a_q, self.linear_a_v]
        self.w_Bs = [self.linear_b_q, self.linear_b_v]


class TinyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyLoRAVisionBackbone()
        self.projection_head_vision = nn.Linear(2, 2)


class TinyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)
        self.projection_head_text = nn.Linear(2, 2)


class TinyRetiZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = TinyVisionModel()
        self.text_model = TinyTextModel()
        self.logit_scale = nn.Parameter(torch.tensor(1.0))


def make_tiny_adapter():
    adapter = RetiZeroAdapter.__new__(RetiZeroAdapter)
    nn.Module.__init__(adapter)
    adapter.retizero = TinyRetiZero()
    adapter.logit_scale = nn.Parameter(torch.tensor(2.0))
    adapter.tokenizer = SimpleNamespace()
    return adapter


def test_configure_cir_finetuning_only_enables_lora_and_projection_heads():
    adapter = make_tiny_adapter()

    trainable_names = adapter.configure_cir_finetuning()

    assert trainable_names
    assert all(
        any(
            token in name
            for token in (
                "linear_a_q",
                "linear_b_q",
                "linear_a_v",
                "linear_b_v",
                "projection_head_vision",
                "projection_head_text",
            )
        )
        for name in trainable_names
    )
    assert not adapter.retizero.vision_model.model.base.weight.requires_grad
    assert not adapter.retizero.text_model.model.weight.requires_grad
    assert not adapter.retizero.logit_scale.requires_grad
    assert not adapter.logit_scale.requires_grad
    assert adapter.visual is adapter.retizero.vision_model


def test_load_checkpoint_restores_full_clip4cir_adapter(tmp_path):
    adapter = make_tiny_adapter()
    expected = {
        key: torch.full_like(value, 3)
        for key, value in adapter.state_dict().items()
    }
    checkpoint = tmp_path / "cir.pt"
    torch.save({"epoch": 7, "RetiZeroAdapter": expected}, checkpoint)

    epoch, metric = adapter.load_checkpoint(checkpoint)

    assert (epoch, metric) == (7, -1)
    assert all(
        torch.equal(adapter.state_dict()[key], value)
        for key, value in expected.items()
    )


def test_load_checkpoint_accepts_legacy_classification_lora(tmp_path):
    adapter = make_tiny_adapter()
    expected = {
        key: torch.full_like(value, 2)
        for key, value in adapter.retizero.vision_model.model.state_dict().items()
    }
    checkpoint = tmp_path / "legacy.pth"
    torch.save(
        {
            "epoch": 4,
            "mean_ACC": 0.8,
            "state_dict": {
                **{f"img_encoder.{key}": value for key, value in expected.items()},
                "classifier.weight": torch.ones(1, 2),
            },
        },
        checkpoint,
    )

    epoch, metric = adapter.load_checkpoint(checkpoint)

    assert (epoch, metric) == (4, 0.8)
    assert all(
        torch.equal(
            adapter.retizero.vision_model.model.state_dict()[key],
            value,
        )
        for key, value in expected.items()
    )
