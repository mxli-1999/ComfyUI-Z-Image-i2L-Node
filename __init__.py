"""
ComfyUI-Z-Image-i2L-Node
=========================
ComfyUI custom-node package that wraps the Z-Image i2L (Image-to-LoRA) pipeline.

Node registrations follow the Comfy-Org/ComfyUI conventions:
  https://github.com/Comfy-Org/ComfyUI
"""

from .nodes import ZImageI2LNode, SaveZImageI2LLoRANode

NODE_CLASS_MAPPINGS = {
    "ZImageI2L": ZImageI2LNode,
    "SaveZImageI2LLoRA": SaveZImageI2LLoRANode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageI2L": "Z-Image i2L (Image to LoRA)",
    "SaveZImageI2LLoRA": "Save Z-Img i2L LoRA",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
