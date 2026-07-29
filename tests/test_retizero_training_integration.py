import sys
from functools import partial
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

import clip_fine_tune
import combiner_train


class FakeRetiZeroAdapter(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.loaded_path = None
        self.configured = False
        self.visual = SimpleNamespace(input_resolution=224, output_dim=512)

    def configure_cir_finetuning(self):
        self.configured = True
        return ["retizero.vision_model.model.linear_a_q.weight"]

    def load_checkpoint(self, checkpoint_path):
        self.loaded_path = checkpoint_path
        return 3, 0.5

    def to(self, target_device):
        self.target_device = str(target_device)
        return self


@pytest.fixture
def fake_retizero_module(monkeypatch):
    module = ModuleType("retizero_adapter")
    module.RetiZeroAdapter = FakeRetiZeroAdapter
    monkeypatch.setitem(sys.modules, "retizero_adapter", module)
    monkeypatch.setitem(sys.modules, "src.retizero_adapter", module)
    return module


def test_finetune_loader_dispatches_retizero_to_its_adapter(fake_retizero_module):
    model, preprocess = clip_fine_tune._load_model_for_finetune(
        "RetiZero",
        {
            "retizero_base_path": "/tmp/RetiZero.pth",
            "force_rgb": True,
        },
    )

    assert isinstance(model, FakeRetiZeroAdapter)
    assert model.model_path == "/tmp/RetiZero.pth"
    assert preprocess is not None


def test_retizero_uses_raw_text_and_model_specific_trainability():
    model = FakeRetiZeroAdapter("/tmp/RetiZero.pth")

    clip_fine_tune._configure_finetune_parameters(
        model,
        "RetiZero",
        "both",
    )

    assert clip_fine_tune._uses_raw_text_inputs("RetiZero")
    assert model.configured
    with pytest.raises(ValueError, match="only supports --encoder both"):
        clip_fine_tune._configure_finetune_parameters(
            model,
            "RetiZero",
            "image",
        )


def test_combiner_loader_restores_cir_checkpoint_on_top_of_base(
    fake_retizero_module,
):
    model = combiner_train._load_retizero_for_combiner(
        {
            "retizero_base_path": "/tmp/RetiZero.pth",
            "clip_model_path": "/tmp/tuned_retizero_best.pt",
        },
        "cpu",
    )

    assert isinstance(model, FakeRetiZeroAdapter)
    assert model.model_path == "/tmp/RetiZero.pth"
    assert model.loaded_path == "/tmp/tuned_retizero_best.pt"
    assert model.target_device == "cpu"


def test_retizero_vit_imports_without_ignored_vendored_models():
    from iden_modules.modeling.LoraRETFound import lora
    from iden_modules.modeling.models_vit import VisionTransformer

    model = VisionTransformer(
        patch_size=16,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model.head = nn.Identity()

    assert callable(lora)
    assert len(model.blocks) == 1
    assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 32)
