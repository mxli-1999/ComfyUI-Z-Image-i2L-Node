"""
nodes.py — ComfyUI node definitions for Z-Image i2L
=====================================================

Node 1: ZImageI2LNode  ("Z-Image i2L (Image to LoRA)")
    Inputs : images (IMAGE), model (MODEL), clip_vision (CLIP_VISION),
             dinov3_model (filename), image2lora_model (filename),
             lora_strength (FLOAT)
    Outputs: model (MODEL) — cloned patcher with LoRA patches applied;
             the raw LoRA dict is also stored in model_options so that
             SaveZImageI2LLoRANode can retrieve it without a separate wire.

Node 2: SaveZImageI2LLoRANode  ("Save Z-Img i2L LoRA")
    Inputs : model (MODEL), filename_prefix (STRING)
    Outputs: (none — output node)
    Action : extracts the LoRA dict stored by ZImageI2LNode and saves it as
             a .safetensors file to ComfyUI's output directory.

Model file layout expected in {ComfyUI}/models/z_image_i2l/ :
    dinov3_image_encoder.safetensors   ← DINOv3-7B encoder weights
    image2lora_style.safetensors       ← Z-Image-i2L style-decoder weights

Download sources:
    DINOv3  : DiffSynth-Studio/General-Image-Encoders  → DINOv3-7B/model.safetensors
    Image2LoRA: DiffSynth-Studio/Z-Image-i2L            → model.safetensors
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

import folder_paths  # ComfyUI built-in

from .i2l_core import center_crop_resize, merge_lora, map_diffsynth_lora_key_to_comfy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: Directory that holds the auxiliary model files (DINOv3 + Image2LoRA).
Z_IMAGE_I2L_MODEL_DIR: str = os.path.join(folder_paths.models_dir, "z_image_i2l")

# ---------------------------------------------------------------------------
# Lazy model loading with a simple module-level cache
# ---------------------------------------------------------------------------

_model_cache: Dict[str, object] = {}


def _load_dinov3(model_path: str):
    """Instantiate and load a DINOv3ImageEncoder from *model_path*."""
    from diffsynth.models.dinov3_image_encoder import DINOv3ImageEncoder  # type: ignore
    from safetensors.torch import load_file

    model = DINOv3ImageEncoder()
    state = load_file(model_path)
    model.load_state_dict(state, strict=False)
    return model.eval()


def _load_image2lora(model_path: str):
    """Instantiate and load a ZImageImage2LoRAModel from *model_path*."""
    from diffsynth.models.z_image_image2lora import ZImageImage2LoRAModel  # type: ignore
    from safetensors.torch import load_file

    model = ZImageImage2LoRAModel()
    state = load_file(model_path)
    model.load_state_dict(state, strict=False)
    return model.eval()


def _cached(path: str, loader):
    """Return a cached model, loading it on first call."""
    if path not in _model_cache:
        _model_cache[path] = loader(path)
    return _model_cache[path]


# ---------------------------------------------------------------------------
# Helper: list .safetensors files in z_image_i2l dir matching a pattern
# ---------------------------------------------------------------------------

def _find_models(pattern_words: List[str], fallback: str) -> List[str]:
    """Return matching .safetensors filenames from Z_IMAGE_I2L_MODEL_DIR."""
    os.makedirs(Z_IMAGE_I2L_MODEL_DIR, exist_ok=True)
    try:
        all_files = [
            f for f in os.listdir(Z_IMAGE_I2L_MODEL_DIR)
            if f.endswith(".safetensors")
        ]
    except OSError:
        all_files = []

    matched = sorted(
        f for f in all_files
        if any(w in f.lower() for w in pattern_words)
    )
    return matched if matched else [fallback]


# ---------------------------------------------------------------------------
# Node 1 — Z-Image i2L (Image to LoRA)
# ---------------------------------------------------------------------------

class ZImageI2LNode:
    """Convert reference style images to LoRA weights and apply to a model.

    Requires two auxiliary model files in ``{ComfyUI}/models/z_image_i2l/``:

    * ``dinov3_image_encoder.safetensors``  — DINOv3-7B encoder
    * ``image2lora_style.safetensors``      — Z-Image i2L style decoder

    The SigLIP2 encoder is supplied via the standard **Load CLIP Vision** node
    (load ``SigLIP2-G384/model.safetensors`` as a CLIP_VISION model).
    """

    @classmethod
    def INPUT_TYPES(cls):
        dinov3_files = _find_models(
            ["dinov3", "dino"], "dinov3_image_encoder.safetensors"
        )
        i2l_files = _find_models(
            ["image2lora", "i2l", "style"], "image2lora_style.safetensors"
        )
        return {
            "required": {
                "images": ("IMAGE",),
                "model": ("MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "dinov3_model": (dinov3_files,),
                "image2lora_model": (i2l_files,),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Strength of the generated LoRA applied to the model.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "encode_and_apply"
    CATEGORY = "Z-Image/i2L"
    DESCRIPTION = (
        "Encode reference style images through SigLIP2 + DINOv3, decode to LoRA "
        "weights via the Z-Image-i2L style decoder, then patch the diffusion model."
    )

    # ------------------------------------------------------------------
    def encode_and_apply(
        self,
        images: torch.Tensor,       # BHWC float32 [0,1]
        model,                       # ComfyUI ModelPatcher
        clip_vision,                 # ComfyUI CLIPVision (SigLIP2-G384)
        dinov3_model: str,
        image2lora_model: str,
        lora_strength: float = 1.0,
    ):
        import comfy.model_management  # type: ignore

        device = comfy.model_management.get_torch_device()
        # Prefer bfloat16 when the model already uses it; fall back to float32.
        dtype: torch.dtype = (
            model.model_dtype()
            if hasattr(model, "model_dtype") and callable(model.model_dtype)
            else torch.bfloat16
        )

        # ── 1. Resolve & validate auxiliary model files ──────────────────
        dinov3_path = os.path.join(Z_IMAGE_I2L_MODEL_DIR, dinov3_model)
        i2l_path = os.path.join(Z_IMAGE_I2L_MODEL_DIR, image2lora_model)

        missing: List[str] = []
        if not os.path.exists(dinov3_path):
            missing.append(
                f"DINOv3 encoder  : {dinov3_path}\n"
                f"  ↳ Download from ModelScope: DiffSynth-Studio/General-Image-Encoders "
                f"→ DINOv3-7B/model.safetensors"
            )
        if not os.path.exists(i2l_path):
            missing.append(
                f"Image2LoRA model: {i2l_path}\n"
                f"  ↳ Download from ModelScope: DiffSynth-Studio/Z-Image-i2L "
                f"→ model.safetensors"
            )
        if missing:
            nl = "\n"
            raise FileNotFoundError(
                f"[Z-Image i2L] Missing model file(s) — place them in:\n"
                f"  {Z_IMAGE_I2L_MODEL_DIR}\n\n"
                + nl.join(missing)
            )

        # ── 2. Load auxiliary models (cached) ────────────────────────────
        dinov3 = _cached(dinov3_path, _load_dinov3)
        image2lora = _cached(i2l_path, _load_image2lora)
        dinov3 = dinov3.to(device=device, dtype=dtype)
        image2lora = image2lora.to(device=device, dtype=dtype)

        # ── 3. Convert IMAGE tensor → pre-processed PIL images ───────────
        # ComfyUI IMAGE: (B, H, W, C), float32, range [0, 1]
        images_pil: List[Image.Image] = []
        for i in range(images.shape[0]):
            arr = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
            pil = center_crop_resize(pil, size=1024)
            images_pil.append(pil)

        # ── 4. Encode through SigLIP2 (via ComfyUI CLIP_VISION) ──────────
        siglip2_embs: List[torch.Tensor] = []
        for pil in images_pil:
            # Re-convert to ComfyUI IMAGE tensor (1, H, W, C) for encode_image()
            arr_f32 = np.array(pil).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(arr_f32).unsqueeze(0)  # (1, 1024, 1024, 3)

            cv_out = clip_vision.encode_image(img_tensor)

            # Extract the global pooled embedding ─────────────────────────
            # CLIPVisionOutput may expose different attribute names depending on
            # the ComfyUI version; try the most common ones in priority order.
            emb: Optional[torch.Tensor] = None
            for attr in ("image_embeds", "pooler_output"):
                val = getattr(cv_out, attr, None)
                if val is not None:
                    emb = val.squeeze(0)
                    break
            if emb is None:
                # Fall back: mean-pool the final hidden-state sequence
                hs = getattr(cv_out, "last_hidden_state", None) or getattr(
                    cv_out, "penultimate_hidden_states", None
                )
                if hs is None:
                    raise RuntimeError(
                        "[Z-Image i2L] Cannot extract embedding from CLIP_VISION output. "
                        "Ensure the loaded CLIP Vision model is SigLIP2-G384."
                    )
                emb = hs.squeeze(0).mean(dim=0)

            siglip2_embs.append(emb.to(device=device, dtype=dtype))

        siglip2_stack = torch.stack(siglip2_embs)  # (N, 1536)

        # ── 5. Encode through DINOv3 ──────────────────────────────────────
        dinov3_embs: List[torch.Tensor] = []
        with torch.no_grad():
            for pil in images_pil:
                emb = dinov3(pil)
                dinov3_embs.append(emb.to(device=device, dtype=dtype))

        dinov3_stack = torch.stack(dinov3_embs)  # (N, 4096)

        # ── 6. Concatenate embeddings (N, 5632) ───────────────────────────
        embeddings = torch.concat([siglip2_stack, dinov3_stack], dim=-1)

        # ── 7. Decode each embedding → per-image LoRA dict ───────────────
        loras: List[Dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for x in embeddings:
                loras.append(image2lora(x=x, residual=None))

        # ── 8. Merge LoRA dicts (average with alpha = 1/N) ───────────────
        lora_dict = merge_lora(loras, alpha=1.0 / len(embeddings))

        # ── 9. Apply LoRA to ComfyUI ModelPatcher ────────────────────────
        new_model = model.clone()

        # Collect model state-dict keys to validate our mapped keys.
        model_sd_keys = (
            set(new_model.model.state_dict().keys())
            if hasattr(new_model, "model")
            else set()
        )

        patches: Dict[str, tuple] = {}
        for key_A in [k for k in lora_dict if ".lora_A." in k]:
            key_B = key_A.replace(".lora_A.", ".lora_B.")

            # Keep on CPU (ComfyUI moves them to GPU during inference).
            lora_A = lora_dict[key_A].cpu().float()  # (rank, in_features)
            lora_B = lora_dict[key_B].cpu().float()  # (out_features, rank)

            # Compute weight delta: Δ = lora_B @ lora_A  →  (out, in)
            delta = lora_B @ lora_A

            comfy_key = map_diffsynth_lora_key_to_comfy(key_A)

            # Only register if the key actually exists in this model.
            # When model_sd_keys is empty (e.g. unloaded model) we accept all keys.
            if not model_sd_keys or comfy_key in model_sd_keys:
                # Store as a 1-tuple: ComfyUI's calculate_weight adds
                # (strength × v[0]) to the original weight tensor.
                patches[comfy_key] = (delta,)

        if patches:
            applied = new_model.add_patches(patches, lora_strength)
            print(
                f"[Z-Image i2L] Applied LoRA to {len(applied)}/{len(patches)} "
                f"layer(s)  (strength={lora_strength:.2f})"
            )
        else:
            print(
                "[Z-Image i2L] Warning: no LoRA patches were applied — the "
                "generated keys did not match any layer in the supplied model. "
                "This is expected when using a non-DiffSynth Z-Image model layout; "
                "you can still save the LoRA via SaveZImageI2LLoRANode."
            )

        # ── 10. Stash raw LoRA dict for the Save node ─────────────────────
        # model_options is deep-copied by ModelPatcher.clone(), so it is safe
        # to store arbitrary data here without affecting the original patcher.
        new_model.model_options["z_image_i2l_lora"] = lora_dict

        return (new_model,)


# ---------------------------------------------------------------------------
# Node 2 — Save Z-Img i2L LoRA
# ---------------------------------------------------------------------------

class SaveZImageI2LLoRANode:
    """Save the LoRA generated by :class:`ZImageI2LNode` to a safetensors file.

    Connect the **model** output of **Z-Image i2L (Image to LoRA)** directly
    to this node.  The file is written to ComfyUI's output directory with an
    auto-generated timestamp suffix.

    The saved file uses DiffSynth key names (``layers.*.lora_A.weight`` etc.)
    and can be loaded back by the original ``ZImagePipeline`` as
    ``positive_only_lora``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": (
                    "STRING",
                    {"default": "z_image_i2l_lora"},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_lora"
    OUTPUT_NODE = True
    CATEGORY = "Z-Image/i2L"
    DESCRIPTION = (
        "Extract the Z-Image i2L LoRA dict from the patched model and write it "
        "to {ComfyUI}/output/<filename_prefix>_<timestamp>.safetensors."
    )

    # ------------------------------------------------------------------
    def save_lora(self, model, filename_prefix: str = "z_image_i2l_lora"):
        from safetensors.torch import save_file  # type: ignore

        # Retrieve the LoRA dict stored by ZImageI2LNode ─────────────────
        lora_dict: Optional[Dict[str, torch.Tensor]] = None
        if hasattr(model, "model_options") and isinstance(model.model_options, dict):
            lora_dict = model.model_options.get("z_image_i2l_lora")

        if lora_dict is None:
            raise ValueError(
                "[Z-Image i2L] No LoRA data found in the supplied model.\n"
                "Make sure to connect this node directly to the MODEL output "
                "of the 'Z-Image i2L (Image to LoRA)' node."
            )

        # Build output path ───────────────────────────────────────────────
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.safetensors"
        output_path = os.path.join(output_dir, filename)

        # Normalise to CPU float32 before saving ──────────────────────────
        lora_save = {k: v.cpu().float().contiguous() for k, v in lora_dict.items()}
        save_file(lora_save, output_path)

        total_params = sum(v.numel() for v in lora_save.values())
        print(
            f"[Z-Image i2L] LoRA saved → {output_path}\n"
            f"              keys: {len(lora_save)}  |  "
            f"total params: {total_params:,}"
        )

        return {}
