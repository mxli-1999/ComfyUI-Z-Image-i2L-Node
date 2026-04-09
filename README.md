# ComfyUI-Z-Image-i2L-Node

ComfyUI custom-node package that wraps the **Z-Image i2L (Image-to-LoRA)** pipeline from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) as native ComfyUI nodes.

Given a batch of reference style images the nodes:

1. Encode each image through **SigLIP2-G384** (via ComfyUI's standard `CLIP_VISION` type) and **DINOv3-7B** (loaded from a local file).
2. Concatenate the two embeddings per image → shape `(N, 5632)`.
3. Pass each embedding through the **Z-Image-i2L style decoder** to obtain per-image LoRA dicts.
4. Merge all per-image LoRAs (average with α = 1/N).
5. Patch the supplied diffusion **MODEL** patcher and return it — ready for a KSampler.

---

## Nodes

### `Z-Image i2L (Image to LoRA)`

| Slot | Name | Type | Description |
|------|------|------|-------------|
| **in** | `images` | `IMAGE` | Batch of reference style images (BHWC, float32 [0,1]) |
| **in** | `model` | `MODEL` | Z-Image diffusion model loaded via *Load Diffusion Model* |
| **in** | `clip_vision` | `CLIP_VISION` | SigLIP2-G384 loaded via *Load CLIP Vision* |
| **in** | `dinov3_model` | dropdown | DINOv3 encoder filename from `models/z_image_i2l/` |
| **in** | `image2lora_model` | dropdown | i2L decoder filename from `models/z_image_i2l/` |
| **in** | `lora_strength` | `FLOAT` | LoRA weight multiplier (default 1.0) |
| **out** | `model` | `MODEL` | Model patcher with LoRA applied; also carries the raw LoRA dict for *Save Z-Img i2L LoRA* |

### `Save Z-Img i2L LoRA`

| Slot | Name | Type | Description |
|------|------|------|-------------|
| **in** | `model` | `MODEL` | Output of *Z-Image i2L* node |
| **in** | `filename_prefix` | `STRING` | Output filename prefix (default `z_image_i2l_lora`) |

Saves `{output_dir}/{filename_prefix}_{timestamp}.safetensors`.  
The file uses DiffSynth key names and is directly compatible with `ZImagePipeline(positive_only_lora=…)`.

---

## Installation

### 1 — Clone into ComfyUI custom nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mxli-1999/ComfyUI-Z-Image-i2L-Node
pip install -r ComfyUI-Z-Image-i2L-Node/requirements.txt
```

### 2 — Download model files

Place the following files under `{ComfyUI}/models/z_image_i2l/`:

| File | Source | ModelScope repo | Path |
|------|--------|-----------------|------|
| `dinov3_image_encoder.safetensors` | DINOv3-7B encoder | `DiffSynth-Studio/General-Image-Encoders` | `DINOv3-7B/model.safetensors` |
| `image2lora_style.safetensors` | Z-Image-i2L decoder | `DiffSynth-Studio/Z-Image-i2L` | `model.safetensors` |

The SigLIP2-G384 model is loaded via ComfyUI's standard `Load CLIP Vision` node:

| File | Source | ModelScope repo | Path |
|------|--------|-----------------|------|
| `siglip2_g384.safetensors` (or any name) | SigLIP2-G384 encoder | `DiffSynth-Studio/General-Image-Encoders` | `SigLIP2-G384/model.safetensors` |

Place the SigLIP2 file anywhere ComfyUI scans for CLIP Vision models (e.g. `models/clip_vision/`).

#### Downloading with ModelScope Python SDK

```python
from modelscope import snapshot_download

# DINOv3 + SigLIP2
snapshot_download(
    "DiffSynth-Studio/General-Image-Encoders",
    local_dir="ComfyUI/models/z_image_i2l",
    allow_file_pattern=["DINOv3-7B/model.safetensors", "SigLIP2-G384/model.safetensors"],
)
# Rename:
#   DINOv3-7B/model.safetensors  → models/z_image_i2l/dinov3_image_encoder.safetensors
#   SigLIP2-G384/model.safetensors → models/clip_vision/siglip2_g384.safetensors

# Image2LoRA decoder
snapshot_download(
    "DiffSynth-Studio/Z-Image-i2L",
    local_dir="ComfyUI/models/z_image_i2l",
    allow_file_pattern="model.safetensors",
)
# Rename: model.safetensors → models/z_image_i2l/image2lora_style.safetensors
```

---

## Workflow example

```
[Load Image (batch)]──────────────────────────────────────────────────────┐
[Load Diffusion Model] ── model ──────────────────────────────────────┐   │
[Load CLIP Vision (SigLIP2)] ── clip_vision ──────────────────────┐   │   │
                                                                   ▼   ▼   ▼
                                               [ Z-Image i2L (Image to LoRA) ]
                                                       │ model (patched)
                                                       ├──► [ KSampler ]
                                                       └──► [ Save Z-Img i2L LoRA ]
```

---

## Package structure

```
ComfyUI-Z-Image-i2L-Node/
├── __init__.py        # NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
├── nodes.py           # ZImageI2LNode, SaveZImageI2LLoRANode
├── i2l_core.py        # center_crop_resize, merge_lora, key mapping — standalone
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Key mapping note

The LoRA weights produced by the style decoder follow DiffSynth key names
(`layers.{i}.attention.to_q.lora_A.weight` etc.).  
When patching a ComfyUI model the node prepends a `diffusion_model.` prefix
(`diffusion_model.layers.{i}.attention.to_q.weight`) which matches the
standard ComfyUI weight layout for Wan/Z-Image diffusion models.

If your Comfy-Org repackaged model uses a different key layout the LoRA
patches will be skipped with a console warning — the raw LoRA dict is still
stored in the model so you can save it with **Save Z-Img i2L LoRA** and
apply it manually.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
