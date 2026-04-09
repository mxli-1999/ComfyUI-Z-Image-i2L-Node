"""
i2l_core.py — Core Z-Image i2L logic
======================================
Standalone from ComfyUI infrastructure.

Provides:
  - center_crop_resize()    : PIL pre-processing (crop + resize to 1024×1024)
  - merge_lora_weight()     : concatenate per-image rank tensors
  - merge_lora()            : average-merge a list of per-image LoRA dicts
  - map_diffsynth_lora_key_to_comfy() : translate DiffSynth LoRA key → ComfyUI weight key
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def center_crop_resize(image: Image.Image, size: int = 1024) -> Image.Image:
    """Center-crop *image* to a square then resize to *size* × *size*.

    This mirrors DiffSynth's ``ImageCropAndResize(height=1024, width=1024)``
    that is applied to every reference image before it is passed to either
    the SigLIP2 or DINOv3 encoder.
    """
    image = image.convert("RGB")
    w, h = image.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    image = image.crop((left, top, left + min_dim, top + min_dim))
    return image.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# LoRA merging (mirrors diffsynth/utils/lora/merge.py)
# ---------------------------------------------------------------------------

def merge_lora_weight(
    tensors_A: List[torch.Tensor],
    tensors_B: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-image LoRA rank tensors into a single merged weight pair.

    Args:
        tensors_A: list of lora_A tensors, each shape ``(rank, in_features)``.
        tensors_B: list of lora_B tensors, each shape ``(out_features, rank)``.

    Returns:
        Tuple ``(lora_A_merged, lora_B_merged)`` where ranks are concatenated:
        ``lora_A_merged`` has shape ``(rank*N, in_features)`` and
        ``lora_B_merged`` has shape ``(out_features, rank*N)``.
    """
    lora_A = torch.concat(tensors_A, dim=0)
    lora_B = torch.concat(tensors_B, dim=1)
    return lora_A, lora_B


def merge_lora(
    loras: List[Dict[str, torch.Tensor]],
    alpha: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Average-merge multiple per-image LoRA dicts into a single LoRA dict.

    Replicates ``diffsynth/utils/lora/merge.py::merge_lora()``.
    Each ``lora_A`` tensor is scaled by *alpha* (= 1/N when N images are used)
    so the combined effect is an average over all input images.

    Args:
        loras: list of LoRA state-dicts, each with keys like
               ``"layers.*.lora_A.weight"`` / ``"layers.*.lora_B.weight"``.
        alpha: scaling factor applied to the merged ``lora_A`` weights.

    Returns:
        Merged LoRA state-dict with the same key layout as the inputs.
    """
    if not loras:
        return {}

    lora_merged: Dict[str, torch.Tensor] = {}
    keys_A = [k for k in loras[0] if ".lora_A." in k]

    for key_A in keys_A:
        key_B = key_A.replace(".lora_A.", ".lora_B.")
        tensors_A = [lora[key_A] for lora in loras]
        tensors_B = [lora[key_B] for lora in loras]
        merged_A, merged_B = merge_lora_weight(tensors_A, tensors_B)
        lora_merged[key_A] = merged_A * alpha
        lora_merged[key_B] = merged_B

    return lora_merged


# ---------------------------------------------------------------------------
# Key mapping: DiffSynth → ComfyUI
# ---------------------------------------------------------------------------

def map_diffsynth_lora_key_to_comfy(
    lora_key: str,
    prefix: str = "diffusion_model",
) -> str:
    """Translate a DiffSynth LoRA key to the corresponding ComfyUI model weight key.

    DiffSynth keys look like::

        layers.0.attention.to_q.lora_A.weight
        layers.0.attention.to_q.lora_B.weight

    ComfyUI wraps the diffusion model parameters under a ``diffusion_model.``
    prefix so the corresponding model state-dict key is::

        diffusion_model.layers.0.attention.to_q.weight

    Args:
        lora_key: DiffSynth LoRA key ending in ``.lora_A.weight`` or
                  ``.lora_B.weight``.
        prefix:   ComfyUI model prefix (default ``"diffusion_model"``).

    Returns:
        ComfyUI model weight key (with ``.weight`` suffix, without lora_A/B part).
    """
    base = lora_key
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{prefix}.{base}.weight"
