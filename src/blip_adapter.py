from types import SimpleNamespace
from typing import List, Optional, Union
from pathlib import Path
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file


class _BLIP2QFormerTextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length].to(input_ids.device)
        position_ids = position_ids.expand_as(input_ids)
        return self.word_embeddings(input_ids) + self.position_embeddings(position_ids)


class _SimpleHashTokenizer:
    def __init__(self, vocab_size: int, pad_token_id: int = 0):
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, return_tensors="pt", padding=True, truncation=True, max_length=77):
        if isinstance(text, str):
            text = [text]
        encoded = []
        for item in text:
            tokens = str(item).lower().replace(".", " ").replace(",", " ").split()
            ids = [2 + (abs(hash(token)) % max(1, self.vocab_size - 2)) for token in tokens]
            if truncation:
                ids = ids[:max_length]
            encoded.append(ids or [1])
        seq_len = max(len(ids) for ids in encoded) if padding else max_length
        if padding:
            seq_len = min(seq_len, max_length)
        input_ids = []
        attention_mask = []
        for ids in encoded:
            ids = ids[:seq_len]
            pad_len = seq_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class BLIPAdapter(nn.Module):
    """
    Adapter that exposes a CLIP-like interface for BLIP/BLIP2.
    It supports two backends:
    - transformers (preferred)
    - lavis (fallback)
    """

    def __init__(
        self,
        model_type: str = "BLIP",
        backend: str = "auto",
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        projection_dim: int = 768,
        input_resolution: int = 224,
        max_text_len: int = 77,
        device: Optional[torch.device] = None,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.model_type = str(model_type).upper()
        self.backend = backend
        self.model_name = model_name
        self.model_path = model_path
        self.projection_dim = int(projection_dim)
        self.input_resolution = int(input_resolution)
        self.max_text_len = int(max_text_len)
        self._device = device
        self.normalize_output = bool(normalize_output)

        self.image_proj: Optional[nn.Linear] = None
        self.text_proj: Optional[nn.Linear] = None

        # Keep CLIP-style log-temperature available for callers.
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        self._tf_model = None
        self._tf_tokenizer = None
        self._tf_vision_model = None
        self._tf_qformer_model = None
        self._tf_query_tokens = None
        self._tf_text_embeddings = None
        self._tf_blip2_qformer_only = False
        self._tf_custom_weights_loaded = False
        self._tf_text_embeddings_loaded = False
        self._tf_hash_tokenizer_active = False
        self._tokenizer_error = None
        self._blip2_text_path = None
        self._projection_heads_loaded = False
        self._projection_heads_random = False
        self._projection_source = "uninitialized"
        self._lavis_model = None
        self._lavis_vis_processor = None
        self._lavis_txt_processor = None

        self._active_backend = self._init_backend()

        self.visual = SimpleNamespace(
            input_resolution=self.input_resolution,
            output_dim=self.projection_dim,
        )

    def _default_transformers_model_name(self) -> str:
        if self.model_type == "BLIP2":
            return "Salesforce/blip2-itm-vit-g"
        return "Salesforce/blip-itm-base-coco"

    def _looks_like_retrieval_checkpoint(self) -> bool:
        if not self.model_path:
            return False
        name = Path(self.model_path).name.lower()
        return any(token in name for token in ("retrieval", "itm", "coco"))

    def _local_model_dir_from_path(self) -> Optional[str]:
        if not self.model_path:
            return None
        path = Path(self.model_path)
        if path.is_file() and (path.parent / "config.json").exists():
            return str(path.parent)
        if path.is_dir() and (path / "config.json").exists():
            return str(path)
        return None

    def _local_model_dir_from_name(self) -> Optional[str]:
        if not self.model_name:
            return None
        path = Path(str(self.model_name))
        if path.is_dir() and (path / "config.json").exists():
            return str(path)
        return None

    def _local_model_dir(self) -> Optional[str]:
        return self._local_model_dir_from_path() or self._local_model_dir_from_name()

    def _transformers_blip_candidates(self) -> List[str]:
        user_name = self.model_name
        retrieval_defaults = [
            "Salesforce/blip-itm-large-coco",
            "Salesforce/blip-itm-base-coco",
        ]
        captioning_default = "Salesforce/blip-image-captioning-base"

        # Retrieval checkpoints should prefer retrieval backbones first.
        if self._looks_like_retrieval_checkpoint():
            if user_name and "captioning" in str(user_name).lower():
                return retrieval_defaults + [str(user_name), captioning_default]
            if user_name:
                return [str(user_name)] + retrieval_defaults + [captioning_default]
            return retrieval_defaults + [captioning_default]

        if user_name:
            return [str(user_name)]
        return [self._default_transformers_model_name(), captioning_default]

    def _transformers_blip2_candidates(self) -> List[str]:
        user_name = self.model_name
        retrieval_default = "Salesforce/blip2-itm-vit-g"
        captioning_default = "Salesforce/blip2-opt-2.7b"

        if self._looks_like_retrieval_checkpoint():
            if user_name and "opt" in str(user_name).lower():
                return [retrieval_default, str(user_name)]
            if user_name:
                return [str(user_name), retrieval_default, captioning_default]
            return [retrieval_default, captioning_default]

        if user_name:
            return [str(user_name)]
        return [self._default_transformers_model_name(), captioning_default]

    def _init_backend(self) -> str:
        candidates = [self.backend]
        if self.backend == "auto":
            # BLIP retrieval checkpoints are usually lavis-native; prefer lavis first if available.
            if self.model_type == "BLIP" and self._looks_like_retrieval_checkpoint():
                candidates = ["lavis", "transformers"]
            else:
                candidates = ["transformers", "lavis"]

        last_err = None
        for cand in candidates:
            if cand == "transformers":
                try:
                    self._init_transformers_backend()
                    return "transformers"
                except Exception as exc:
                    last_err = exc
            elif cand == "lavis":
                try:
                    self._init_lavis_backend()
                    return "lavis"
                except Exception as exc:
                    last_err = exc
            else:
                raise ValueError(f"Unsupported BLIP backend: {cand}")

        raise RuntimeError(
            "Failed to initialize BLIP backend. "
            "Install transformers or lavis. "
            f"Last error: {last_err}"
        )

    def _init_transformers_backend(self):
        if self.model_type == "BLIP2":
            from transformers import AutoTokenizer, Blip2Config, Blip2QFormerModel, Blip2VisionModel

            last_err = None
            selected_name = None
            for model_name in self._transformers_blip2_candidates():
                try:
                    config_source = self._local_model_dir() or model_name
                    config = Blip2Config.from_pretrained(config_source, local_files_only=True)
                    self.input_resolution = int(config.vision_config.image_size)
                    self._tf_vision_model = Blip2VisionModel(config.vision_config)
                    self._tf_qformer_model = Blip2QFormerModel(config.qformer_config)
                    self._tf_query_tokens = nn.Parameter(
                        torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
                    )
                    tokenizer_source = self._local_model_dir() or model_name
                    architectures = [str(item) for item in getattr(config, "architectures", []) or []]
                    has_retrieval_text_path = bool(
                        getattr(config.qformer_config, "use_qformer_text_input", False)
                        or any("ImageTextRetrieval" in item for item in architectures)
                    )
                    self._tf_tokenizer = self._load_blip2_tokenizer(tokenizer_source, config)
                    if not has_retrieval_text_path:
                        generative_warning = (
                            "Selected BLIP2 checkpoint is generative-only in this transformers version; "
                            "using its local tokenizer with tuned Q-Former text embeddings as a fallback "
                            "so training can run, but this is not an official CLIP-like retrieval text path."
                        )
                        if self._tokenizer_error:
                            self._tokenizer_error = f"{generative_warning} Tokenizer load notes: {self._tokenizer_error}"
                        else:
                            self._tokenizer_error = generative_warning
                    text_vocab_size = max(
                        int(getattr(config.qformer_config, "vocab_size", 0) or 0),
                        len(self._tf_tokenizer) if self._tf_tokenizer is not None else 0,
                    )
                    if text_vocab_size <= 0:
                        text_vocab_size = 30522
                    self._tf_text_embeddings = _BLIP2QFormerTextEmbeddings(
                        text_vocab_size,
                        config.qformer_config.hidden_size,
                        config.qformer_config.max_position_embeddings,
                    )
                    self._tf_model = self._tf_qformer_model
                    self._tf_blip2_qformer_only = True
                    selected_name = model_name
                    break
                except Exception as exc:
                    last_err = exc

            if self._tf_model is None:
                raise RuntimeError(
                    f"Failed to init transformers BLIP2 backbone from candidates: "
                    f"{self._transformers_blip2_candidates()}. Last error: {last_err}"
                ) from last_err

            print(f"BLIPAdapter transformers backbone: {selected_name}")

            checkpoint = self._load_checkpoint(self.model_path) if self.model_path else None
            state_dict = self._extract_state_dict_from_checkpoint(checkpoint) if checkpoint is not None else None
            self._init_blip2_projection_heads(state_dict, config.qformer_config.hidden_size)
            if state_dict is not None:
                self._load_blip2_qformer_components(state_dict)
                self._tf_custom_weights_loaded = True
        else:
            from transformers import AutoTokenizer, BlipForImageTextRetrieval, BlipModel

            last_err = None
            selected_name = None
            for model_name in self._transformers_blip_candidates():
                try:
                    model_source = self._local_model_dir() or model_name
                    try:
                        self._tf_model = BlipForImageTextRetrieval.from_pretrained(
                            model_source,
                            local_files_only=True,
                        )
                    except Exception:
                        self._tf_model = BlipModel.from_pretrained(model_source, local_files_only=True)
                    self._tf_tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
                    if hasattr(self._tf_model, "vision_proj") and hasattr(self._tf_model, "text_proj"):
                        self.image_proj = self._tf_model.vision_proj
                        self.text_proj = self._tf_model.text_proj
                        self.projection_dim = int(self.image_proj.out_features)
                        self._projection_heads_loaded = True
                        self._projection_heads_random = False
                        self._projection_source = "BLIP checkpoint retrieval projection heads"
                    else:
                        hidden_size = int(getattr(self._tf_model.config, "projection_dim", self.projection_dim))
                        self._init_random_projection_heads(hidden_size, hidden_size, "BLIP transformers default")
                    # NOTE: _tf_custom_weights_loaded is set later ONLY when fine-tuned weights
                    # from model_path are successfully loaded, NOT for local base model cache.
                    selected_name = model_name
                    break
                except Exception as exc:
                    last_err = exc

            if self._tf_model is None or self._tf_tokenizer is None:
                raise RuntimeError(
                    f"Failed to init transformers BLIP backbone from candidates: {self._transformers_blip_candidates()}"
                ) from last_err

            print(f"BLIPAdapter transformers backbone: {selected_name}")

        if self.model_path and not self._tf_custom_weights_loaded:
            checkpoint = self._load_checkpoint(self.model_path)
            state_dict = self._extract_state_dict_from_checkpoint(checkpoint)
            if state_dict is not None:
                # --- [BUG FIX] Normalize keys: strip _tf_model. prefix and remap BLIPAdapter direct-child names ---
                # The saved BLIPAdapter state_dict has keys like:
                #   _tf_model.vision_proj.weight  (via _tf_model submodule)
                #   image_proj.weight             (via BLIPAdapter direct child, same tensor as above)
                # But self._tf_model.state_dict() expects:
                #   vision_proj.weight            (transformers native naming)
                # This mismatch caused ALL fine-tuned weights to be silently dropped.
                normalized_state = {}
                renamed_count = 0
                for key, value in state_dict.items():
                    # Strip _tf_model. prefix (from BLIPAdapter internal module hierarchy)
                    if key.startswith("_tf_model."):
                        key = key[len("_tf_model."):]
                        renamed_count += 1
                    # Remap BLIPAdapter direct-child names to transformers model names
                    if key.startswith("image_proj."):
                        key = "vision_proj." + key[len("image_proj."):]
                        renamed_count += 1
                    normalized_state[key] = value
                if renamed_count > 0:
                    print(
                        f"BLIPAdapter key normalization: {renamed_count} keys remapped "
                        f"(stripped _tf_model. prefix and/or image_proj→vision_proj rename)"
                    )
                # --- End BUG FIX ---

                model_state = self._tf_model.state_dict()
                compatible_state_dict = {}
                skipped_mismatch = 0
                for key, value in normalized_state.items():
                    if key not in model_state:
                        compatible_state_dict[key] = value
                        continue
                    if hasattr(value, "shape") and value.shape != model_state[key].shape:
                        skipped_mismatch += 1
                        continue
                    compatible_state_dict[key] = value

                missing, unexpected = self._tf_model.load_state_dict(compatible_state_dict, strict=False)
                loaded_count = len(compatible_state_dict) - len(unexpected)
                print(
                    f"BLIPAdapter custom weights loaded from {self.model_path}. "
                    f"loaded_keys={loaded_count}, missing={len(missing)}, unexpected={len(unexpected)}, skipped_mismatch={skipped_mismatch}"
                )
                if loaded_count == 0 and len(unexpected) > 0:
                    print(
                        "WARNING: No custom weights were loaded into the BLIP backbone! "
                        "The fine-tuned checkpoint may have an unexpected key structure. "
                        "Sample unexpected keys: " + ", ".join(list(unexpected)[:5])
                    )
                self._tf_custom_weights_loaded = True

    def _load_blip2_tokenizer(self, tokenizer_source: str, config):
        from transformers import AutoTokenizer

        candidates = [tokenizer_source]
        if self.model_name:
            candidates.append(str(self.model_name))
        if self.model_type == "BLIP2" and getattr(config.qformer_config, "vocab_size", None) in (30522, 30523):
            candidates.append("bert-base-uncased")

        errors = []
        for source in dict.fromkeys(candidates):
            for use_fast in (True, False):
                try:
                    return AutoTokenizer.from_pretrained(
                        source,
                        local_files_only=True,
                        use_fast=use_fast,
                    )
                except Exception as exc:
                    errors.append(f"{source} (use_fast={use_fast}): {exc}")

        self._tokenizer_error = " | ".join(errors)
        if os.environ.get("CLIP4CIR_ALLOW_HASH_TOKENIZER") == "1":
            fallback_vocab_size = int(getattr(config.qformer_config, "vocab_size", 0) or 0)
            fallback_vocab_size = fallback_vocab_size or int(getattr(config.text_config, "vocab_size", 32128) or 32128)
            self._tf_hash_tokenizer_active = True
            print(
                "BLIPAdapter tokenizer fallback: using simple hash tokenizer because "
                "CLIP4CIR_ALLOW_HASH_TOKENIZER=1. This is diagnostic-only and not a fair retrieval baseline."
            )
            return _SimpleHashTokenizer(fallback_vocab_size)

        print(
            "BLIPAdapter tokenizer unavailable. Text encoding will fail until a local tokenizer is provided. "
            f"Last errors: {self._tokenizer_error}"
        )
        return None

    def _init_random_projection_heads(self, image_in_dim: int, text_in_dim: int, source: str):
        self.image_proj = nn.Linear(int(image_in_dim), self.projection_dim)
        self.text_proj = nn.Linear(int(text_in_dim), self.projection_dim)
        self._projection_heads_loaded = False
        self._projection_heads_random = True
        self._projection_source = f"random:{source}"

    def _init_pretrained_projection_heads(
        self,
        vision_weight: torch.Tensor,
        vision_bias: Optional[torch.Tensor],
        text_weight: torch.Tensor,
        text_bias: Optional[torch.Tensor],
        source: str,
    ):
        image_out_dim, image_in_dim = vision_weight.shape
        text_out_dim, text_in_dim = text_weight.shape
        if image_out_dim != text_out_dim:
            raise RuntimeError(
                f"BLIP2 projection head output dims differ: image={image_out_dim}, text={text_out_dim}"
            )

        self.projection_dim = int(image_out_dim)
        self.image_proj = nn.Linear(int(image_in_dim), self.projection_dim, bias=vision_bias is not None)
        self.text_proj = nn.Linear(int(text_in_dim), self.projection_dim, bias=text_bias is not None)
        with torch.no_grad():
            self.image_proj.weight.copy_(vision_weight.float())
            self.text_proj.weight.copy_(text_weight.float())
            if vision_bias is not None:
                self.image_proj.bias.copy_(vision_bias.float())
            if text_bias is not None:
                self.text_proj.bias.copy_(text_bias.float())
        self._projection_heads_loaded = True
        self._projection_heads_random = False
        self._projection_source = source

    @staticmethod
    def _first_existing_tensor(state_dict: Optional[dict], keys: List[str]) -> Optional[torch.Tensor]:
        if state_dict is None:
            return None
        for key in keys:
            value = state_dict.get(key)
            if torch.is_tensor(value):
                return value
        return None

    def _init_blip2_projection_heads(self, state_dict: Optional[dict], qformer_hidden_size: int):
        vision_weight = self._first_existing_tensor(
            state_dict,
            ["vision_projection.weight", "image_proj.weight"],
        )
        text_weight = self._first_existing_tensor(
            state_dict,
            ["text_projection.weight", "text_proj.weight"],
        )

        if vision_weight is not None and text_weight is not None:
            self._init_pretrained_projection_heads(
                vision_weight=vision_weight,
                vision_bias=self._first_existing_tensor(state_dict, ["vision_projection.bias", "image_proj.bias"]),
                text_weight=text_weight,
                text_bias=self._first_existing_tensor(state_dict, ["text_projection.bias", "text_proj.bias"]),
                source="checkpoint projection heads",
            )
            print(
                "BLIPAdapter BLIP2 projection heads loaded from checkpoint. "
                f"output_dim={self.projection_dim}"
            )
            return

        self._init_random_projection_heads(qformer_hidden_size, qformer_hidden_size, "missing BLIP2 retrieval projection heads")
        print(
            "WARNING: BLIPAdapter BLIP2 projection heads are randomly initialized. "
            "Frozen BLIP2+Combiner results from this model should be treated as a random-head baseline."
        )

    @staticmethod
    def _load_checkpoint(model_path: str):
        path = Path(model_path)
        if path.is_dir():
            safetensors_index = path / "model.safetensors.index.json"
            if safetensors_index.exists():
                return BLIPAdapter._load_sharded_safetensors_subset(
                    path,
                    safetensors_index,
                    prefixes=(
                        "vision_model.",
                        "qformer.",
                        "embeddings.",
                        "vision_projection.",
                        "text_projection.",
                        "image_proj.",
                        "text_proj.",
                    ),
                    exact_keys=("query_tokens",),
                )
            single_safetensors = path / "model.safetensors"
            if single_safetensors.exists():
                return safetensors_load_file(str(single_safetensors), device="cpu")
            raise FileNotFoundError(f"No supported model weights found in {model_path}")

        if str(path).lower().endswith(".safetensors"):
            return safetensors_load_file(str(path), device="cpu")
        return torch.load(str(path), map_location="cpu")

    @staticmethod
    def _load_sharded_safetensors_subset(root: Path, index_path: Path, prefixes, exact_keys):
        index_data = json.loads(index_path.read_text())
        weight_map = index_data.get("weight_map", {})
        wanted = {
            key: shard
            for key, shard in weight_map.items()
            if key in exact_keys or any(key.startswith(prefix) for prefix in prefixes)
        }
        state_dict = {}
        for shard in sorted(set(wanted.values())):
            shard_keys = [key for key, value in wanted.items() if value == shard]
            shard_path = root / shard
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for key in shard_keys:
                    state_dict[key] = handle.get_tensor(key)
        print(
            f"BLIPAdapter loaded {len(state_dict)}/{len(weight_map)} tensors "
            f"from sharded safetensors index {index_path.name}"
        )
        return state_dict

    @staticmethod
    def _strip_prefix_state_dict(state_dict: dict, prefix: str) -> dict:
        prefix = f"{prefix}."
        return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}

    def _load_blip2_qformer_components(self, state_dict: dict):
        vision_state_dict = self._strip_prefix_state_dict(state_dict, "vision_model")
        if not vision_state_dict:
            vision_state_dict = self._strip_prefix_state_dict(state_dict, "_tf_vision_model")
        qformer_state_dict = self._strip_prefix_state_dict(state_dict, "qformer")
        if not qformer_state_dict:
            qformer_state_dict = self._strip_prefix_state_dict(state_dict, "_tf_qformer_model")

        vision_missing, vision_unexpected = self._tf_vision_model.load_state_dict(vision_state_dict, strict=False)
        qformer_missing, qformer_unexpected = self._tf_qformer_model.load_state_dict(qformer_state_dict, strict=False)

        with torch.no_grad():
            if "query_tokens" in state_dict and state_dict["query_tokens"].shape == self._tf_query_tokens.shape:
                self._tf_query_tokens.copy_(state_dict["query_tokens"])
            elif "_tf_query_tokens" in state_dict and state_dict["_tf_query_tokens"].shape == self._tf_query_tokens.shape:
                self._tf_query_tokens.copy_(state_dict["_tf_query_tokens"])

            word_weight = self._first_existing_tensor(
                state_dict,
                ["embeddings.word_embeddings.weight", "_tf_text_embeddings.word_embeddings.weight"],
            )
            pos_weight = self._first_existing_tensor(
                state_dict,
                ["embeddings.position_embeddings.weight", "_tf_text_embeddings.position_embeddings.weight"],
            )
            if word_weight is not None:
                max_positions = (
                    int(pos_weight.shape[0])
                    if pos_weight is not None
                    else self._tf_text_embeddings.position_embeddings.num_embeddings
                )
                if (
                    self._tf_text_embeddings.word_embeddings.weight.shape != word_weight.shape
                    or self._tf_text_embeddings.position_embeddings.num_embeddings != max_positions
                ):
                    self._tf_text_embeddings = _BLIP2QFormerTextEmbeddings(
                        int(word_weight.shape[0]),
                        int(word_weight.shape[1]),
                        max_positions,
                    )
                self._tf_text_embeddings.word_embeddings.weight.copy_(word_weight)
                self._tf_text_embeddings_loaded = True
            if pos_weight is not None:
                self._tf_text_embeddings.position_embeddings.weight.copy_(pos_weight)
                self._tf_text_embeddings_loaded = True

        print(
            f"BLIPAdapter BLIP2 Q-Former weights loaded from {self.model_path}. "
            f"vision_missing={len(vision_missing)}, vision_unexpected={len(vision_unexpected)}, "
            f"qformer_missing={len(qformer_missing)}, qformer_unexpected={len(qformer_unexpected)}, "
            f"text_embeddings_loaded={self._tf_text_embeddings_loaded}"
        )

    @staticmethod
    def _extract_state_dict_from_checkpoint(checkpoint):
        state_dict = checkpoint
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "CLIP" in state_dict and isinstance(state_dict["CLIP"], dict):
            state_dict = state_dict["CLIP"]
        elif isinstance(state_dict, dict) and "BLIPAdapter" in state_dict and isinstance(state_dict["BLIPAdapter"], dict):
            state_dict = state_dict["BLIPAdapter"]
        elif isinstance(state_dict, dict):
            dict_values = [v for v in state_dict.values() if isinstance(v, dict)]
            if len(dict_values) == 1:
                state_dict = dict_values[0]

        if not isinstance(state_dict, dict):
            return None

        # Remove common wrappers.
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        if any(k.startswith("clip_model.") for k in state_dict.keys()):
            state_dict = {k.replace("clip_model.", "", 1): v for k, v in state_dict.items()}

        # Keep tensor entries only; skip optimizer states or metadata.
        state_dict = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
        return state_dict

    def _init_lavis_backend(self):
        from lavis.models import load_model_and_preprocess

        if self.model_type == "BLIP2":
            model_name = "blip2_feature_extractor"
            model_variant = "pretrain"
        else:
            model_name = "blip_feature_extractor"
            model_variant = "base"

        lavis_device = "cpu"
        if self._device is not None and str(self._device).startswith("cuda"):
            lavis_device = str(self._device)

        model, vis_processors, txt_processors = load_model_and_preprocess(
            name=model_name,
            model_type=model_variant,
            is_eval=False,
            device=lavis_device,
        )
        self._lavis_model = model
        self._lavis_vis_processor = vis_processors["eval"]
        self._lavis_txt_processor = txt_processors["eval"]

    def _get_runtime_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        return next(self.parameters()).device

    def _ensure_proj(self, in_dim: int, branch: str, device: torch.device):
        if branch == "image":
            if self.image_proj is None:
                self.image_proj = nn.Linear(in_dim, self.projection_dim).to(device)
                self._projection_heads_random = True
                self._projection_source = "lazy random image projection"
            elif next(self.image_proj.parameters()).device != device:
                self.image_proj = self.image_proj.to(device)
        else:
            if self.text_proj is None:
                self.text_proj = nn.Linear(in_dim, self.projection_dim).to(device)
                self._projection_heads_random = True
                self._projection_source = "lazy random text projection"
            elif next(self.text_proj.parameters()).device != device:
                self.text_proj = self.text_proj.to(device)

    def initialize_projection_heads(self, device: Optional[torch.device] = None):
        """Public hook used before optimizer construction."""
        if device is None:
            device = self._get_runtime_device()
        if self.image_proj is not None:
            self.image_proj = self.image_proj.to(device)
        if self.text_proj is not None:
            self.text_proj = self.text_proj.to(device)
        if self.image_proj is None or self.text_proj is None:
            raise RuntimeError(
                "BLIPAdapter projection heads are not initialized. "
                "Run a supported backend initialization path before creating the optimizer."
            )
        self.visual.output_dim = self.projection_dim
        return self.diagnostic_summary()

    def diagnostic_summary(self) -> dict:
        return {
            "model_type": self.model_type,
            "backend": self._active_backend,
            "projection_dim": self.projection_dim,
            "image_proj": self.image_proj is not None,
            "text_proj": self.text_proj is not None,
            "projection_heads_loaded": self._projection_heads_loaded,
            "projection_heads_random": self._projection_heads_random,
            "projection_source": self._projection_source,
            "blip2_text_embeddings_loaded": self._tf_text_embeddings_loaded,
            "hash_tokenizer_active": self._tf_hash_tokenizer_active,
            "tokenizer_available": self._tf_tokenizer is not None,
            "tokenizer_error": self._tokenizer_error,
            "blip2_text_path": self._blip2_text_path,
        }

    def _normalize_text_input(self, text: Union[List[str], torch.Tensor, dict]) -> dict:
        device = self._get_runtime_device()
        if isinstance(text, dict):
            return {k: v.to(device) for k, v in text.items()}
        if isinstance(text, torch.Tensor):
            return {
                "input_ids": text.to(device),
                "attention_mask": (text != 0).long().to(device),
            }
        if not isinstance(text, list):
            raise TypeError(f"Unsupported text input type: {type(text)}")
        if self._tf_tokenizer is None:
            raise RuntimeError(
                "BLIPAdapter has no tokenizer for raw text input. "
                f"Tokenizer errors: {self._tokenizer_error}"
            )

        encoded = self._tf_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
        )
        text_inputs = {k: v.to(device) for k, v in encoded.items()}
        text_inputs.pop("token_type_ids", None)
        return text_inputs

    @staticmethod
    def _extract_sequence_tensor(outputs) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return outputs
        value = getattr(outputs, "last_hidden_state", None)
        if torch.is_tensor(value):
            return value
        if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
            return outputs[0]
        raise TypeError(f"Unsupported sequence output type: {type(outputs)}")

    @staticmethod
    def _extract_feature_tensor(outputs) -> torch.Tensor:
        if torch.is_tensor(outputs):
            if outputs.dim() == 3:
                return outputs[:, 0, :]
            return outputs

        for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
            value = getattr(outputs, attr, None)
            if torch.is_tensor(value):
                if value.dim() == 3:
                    return value[:, 0, :]
                return value

        logits = getattr(outputs, "logits", None)
        if torch.is_tensor(logits):
            if logits.dim() == 3:
                return logits[:, -1, :]
            return logits

        if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
            first = outputs[0]
            if first.dim() == 3:
                return first[:, 0, :]
            return first

        raise TypeError(f"Unsupported feature output type: {type(outputs)}")

    def _extract_blip2_qformer_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if self._tf_blip2_qformer_only:
            vision_outputs = self._tf_vision_model(pixel_values=image, return_dict=True)
            image_embeds = vision_outputs.last_hidden_state
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
            query_tokens = self._tf_query_tokens.expand(image_embeds.shape[0], -1, -1)
            qformer_outputs = self._tf_qformer_model(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            return self._extract_sequence_tensor(qformer_outputs).mean(dim=1)

        if hasattr(self._tf_model, "get_qformer_features"):
            return self._extract_sequence_tensor(self._tf_model.get_qformer_features(pixel_values=image)).mean(dim=1)

        if not all(hasattr(self._tf_model, attr) for attr in ("vision_model", "qformer", "query_tokens")):
            return self._extract_feature_tensor(self._tf_model.get_image_features(pixel_values=image))

        vision_outputs = self._tf_model.vision_model(pixel_values=image, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = self._tf_model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        qformer_outputs = self._tf_model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        return self._extract_sequence_tensor(qformer_outputs).mean(dim=1)

    def _extract_blip_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if hasattr(self._tf_model, "get_image_features"):
            return self._extract_feature_tensor(self._tf_model.get_image_features(pixel_values=image))

        if hasattr(self._tf_model, "vision_model"):
            vision_outputs = self._tf_model.vision_model(
                pixel_values=image,
                return_dict=True,
                interpolate_pos_encoding=True,
            )
            return self._extract_sequence_tensor(vision_outputs)[:, 0, :]

        raise RuntimeError("The selected BLIP transformers model does not expose an image feature path.")

    def _extract_blip_text_tensor(self, text_inputs: dict) -> torch.Tensor:
        if hasattr(self._tf_model, "get_text_features"):
            return self._extract_feature_tensor(self._tf_model.get_text_features(**text_inputs))

        if hasattr(self._tf_model, "text_encoder"):
            text_outputs = self._tf_model.text_encoder(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
                return_dict=True,
            )
            return self._extract_sequence_tensor(text_outputs)[:, 0, :]

        raise RuntimeError("The selected BLIP transformers model does not expose a text feature path.")

    def _run_blip2_qformer_text_only(self, query_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        """Run Q-Former on text tokens without activating image cross-attention blocks."""
        qformer = self._tf_qformer_model if self._tf_blip2_qformer_only else self._tf_model.qformer
        embedding_output = qformer.layernorm(query_embeds)
        embedding_output = qformer.dropout(embedding_output)

        input_shape = embedding_output.size()[:-1]
        device = embedding_output.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=device)
        extended_attention_mask = qformer.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = qformer.get_head_mask(None, qformer.config.num_hidden_layers)

        try:
            encoder_outputs = qformer.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                query_length=0,
            )
            self._blip2_text_path = "qformer_text_only"
        except AttributeError as exc:
            if "intermediate" not in str(exc):
                raise
            # Older transformers exposes BLIP2 Q-Former as query-only layers. Use a zero
            # encoder state so cross-attention layers can execute without image leakage.
            encoder_hidden_size = int(getattr(qformer.config, "encoder_hidden_size", embedding_output.shape[-1]))
            dummy_encoder = torch.zeros(
                embedding_output.shape[0],
                1,
                encoder_hidden_size,
                dtype=embedding_output.dtype,
                device=embedding_output.device,
            )
            dummy_encoder_mask = torch.ones(
                dummy_encoder.size()[:-1],
                dtype=torch.long,
                device=embedding_output.device,
            )
            dummy_encoder_mask = qformer.invert_attention_mask(dummy_encoder_mask)
            encoder_outputs = qformer.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=dummy_encoder,
                encoder_attention_mask=dummy_encoder_mask,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                query_length=embedding_output.shape[1],
            )
            self._blip2_text_path = "qformer_query_with_zero_encoder_fallback"
        return encoder_outputs.last_hidden_state

    def _extract_blip2_qformer_text_tensor(self, text_inputs: dict) -> torch.Tensor:
        if self._tf_blip2_qformer_only:
            if self._tf_tokenizer is None:
                raise RuntimeError(
                    "BLIP2 text tokenizer is unavailable. Provide a local tokenizer via --blip-model-name/model path "
                    "or install/cache the required tokenizer. "
                    f"Tokenizer errors: {self._tokenizer_error}"
                )
            if not self._tf_text_embeddings_loaded:
                raise RuntimeError(
                    "BLIP2 Q-Former text embeddings were not loaded from a retrieval checkpoint. "
                    "This is typical for generative BLIP2-FLAN checkpoints; treat it as a generative model unless "
                    "you provide a tuned/retrieval checkpoint with text embeddings."
                )
            if self._tf_hash_tokenizer_active:
                raise RuntimeError(
                    "Simple hash tokenizer is active. Refusing to report this as a fair BLIP2 retrieval run."
                )
            query_embeds = self._tf_text_embeddings(text_inputs["input_ids"])
            attention_mask = text_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones(query_embeds.size()[:-1], dtype=torch.long, device=query_embeds.device)
            return self._run_blip2_qformer_text_only(query_embeds, attention_mask)[:, 0, :]

        if hasattr(self._tf_model, "get_text_features"):
            return self._extract_feature_tensor(self._tf_model.get_text_features(**text_inputs))

        if not all(hasattr(self._tf_model, attr) for attr in ("embeddings", "qformer")):
            raise RuntimeError("The selected BLIP2 transformers model does not expose a text Q-Former path.")

        query_embeds = self._tf_model.embeddings(input_ids=text_inputs["input_ids"])
        return self._run_blip2_qformer_text_only(query_embeds, text_inputs.get("attention_mask"))[:, 0, :]

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        device = self._get_runtime_device()
        image = image.to(device)

        if image.dtype != torch.float32 and image.dtype != torch.float16:
            image = image.float()

        if image.shape[-1] != self.input_resolution or image.shape[-2] != self.input_resolution:
            image = F.interpolate(
                image,
                size=(self.input_resolution, self.input_resolution),
                mode="bicubic",
                align_corners=False,
            )

        if self._active_backend == "transformers":
            if self.model_type == "BLIP2":
                image_tensor = self._extract_blip2_qformer_image_tensor(image)
            else:
                image_tensor = self._extract_blip_image_tensor(image)
            self._ensure_proj(image_tensor.shape[-1], "image", image_tensor.device)
            image_features = self.image_proj(image_tensor)
            if self.normalize_output:
                return F.normalize(image_features, dim=-1)
            return image_features

        # lavis fallback path
        from torchvision.transforms.functional import to_pil_image

        processed = []
        for img in image:
            pil = to_pil_image(img.detach().cpu())
            processed.append(self._lavis_vis_processor(pil))
        pixel_values = torch.stack(processed).to(device)

        feats = self._lavis_model.extract_features({"image": pixel_values}, mode="image")
        image_embed = feats.image_embeds[:, 0, :]
        self._ensure_proj(image_embed.shape[-1], "image", image_embed.device)
        image_features = self.image_proj(image_embed)
        if self.normalize_output:
            return F.normalize(image_features, dim=-1)
        return image_features

    def encode_text(self, text: Union[List[str], torch.Tensor, dict]) -> torch.Tensor:
        device = self._get_runtime_device()

        if self._active_backend == "transformers":
            text_inputs = self._normalize_text_input(text)
            if self.model_type == "BLIP2":
                text_tensor = self._extract_blip2_qformer_text_tensor(text_inputs)
            else:
                text_tensor = self._extract_blip_text_tensor(text_inputs)
            self._ensure_proj(text_tensor.shape[-1], "text", text_tensor.device)
            text_features = self.text_proj(text_tensor)
            if self.normalize_output:
                return F.normalize(text_features, dim=-1)
            return text_features

        # lavis fallback path
        if isinstance(text, torch.Tensor):
            text = text.detach().cpu().tolist()
        if isinstance(text, dict):
            text = text.get("text_input", [])

        processed = []
        for item in text:
            processed.append(self._lavis_txt_processor(str(item)))

        feats = self._lavis_model.extract_features({"text_input": processed}, mode="text")
        text_embed = feats.text_embeds[:, 0, :].to(device)
        self._ensure_proj(text_embed.shape[-1], "text", text_embed.device)
        text_features = self.text_proj(text_embed)
        if self.normalize_output:
            return F.normalize(text_features, dim=-1)
        return text_features

    def forward(self, x, mode=None):
        if mode == "image":
            return self.encode_image(x)
        if mode == "text":
            return self.encode_text(x)
        raise ValueError("BLIPAdapter.forward requires mode in {'image', 'text'}")
