import argparse
import json
from pathlib import Path

import torch

try:
    from src.blip_adapter import BLIPAdapter
except ImportError:
    from blip_adapter import BLIPAdapter


def _optimizer_coverage(model: torch.nn.Module):
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW([param for _, param in trainable], lr=1e-6)
    optimizer_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    missing = [name for name, param in trainable if id(param) not in optimizer_ids]
    return {
        "trainable_tensors": len(trainable),
        "trainable_scalars": sum(param.numel() for _, param in trainable),
        "optimizer_covered_tensors": len(trainable) - len(missing),
        "optimizer_missing": missing,
    }


def main():
    parser = argparse.ArgumentParser(description="Static BLIP/BLIP2 adapter sanity check.")
    parser.add_argument("--model-type", default="BLIP2")
    parser.add_argument("--backend", default="transformers")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--projection-dim", type=int, default=768)
    parser.add_argument("--input-resolution", type=int, default=224)
    parser.add_argument("--max-text-len", type=int, default=77)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-forward", action="store_true")
    args = parser.parse_args()

    model = BLIPAdapter(
        model_type=args.model_type,
        backend=args.backend,
        model_name=args.model_name,
        model_path=args.model_path,
        projection_dim=args.projection_dim,
        input_resolution=args.input_resolution,
        max_text_len=args.max_text_len,
        device=torch.device(args.device),
    ).to(args.device)

    summary = model.initialize_projection_heads(device=torch.device(args.device))
    result = {
        "model_path_exists": Path(args.model_path).exists() if args.model_path else None,
        "adapter": summary,
        "optimizer": _optimizer_coverage(model),
    }

    if args.run_forward:
        model.eval()
        with torch.no_grad():
            image = torch.zeros(
                2,
                3,
                model.visual.input_resolution,
                model.visual.input_resolution,
                device=args.device,
            )
            image_features = model.encode_image(image)
            result["image_forward"] = {
                "shape": list(image_features.shape),
                "norm_mean": float(image_features.norm(dim=-1).mean().item()),
            }
            captions = ["retinal hemorrhage near the macula", "hard exudates in the temporal retina"]
            text_features = model.encode_text(captions)
            result["text_forward"] = {
                "shape": list(text_features.shape),
                "norm_mean": float(text_features.norm(dim=-1).mean().item()),
                "paired_cosine": float((image_features * text_features).sum(dim=-1).mean().item()),
            }

    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
