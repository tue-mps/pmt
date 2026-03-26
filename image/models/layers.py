# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import math
from typing import Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers.pytorch_utils import compile_compatible_method_lru_cache
from timm.layers import LayerNorm2d

from models.utils import (
    apply_rotary_pos_emb,
    augment_patches_center_coordinates,
    get_patches_center_coordinates,
)


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x


class Mask2FormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    @compile_compatible_method_lru_cache(maxsize=1)
    def forward(
        self,
        shape: torch.Size,
        device: Union[torch.device, str],
        dtype: torch.dtype,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if mask is None:
            mask = torch.zeros((shape[0], shape[2], shape[3]), device=device, dtype=torch.bool)
        not_mask = (~mask).to(dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=device).to(dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ---------------------------------------------------------------------------
# DINOv3 layers
# ---------------------------------------------------------------------------

class DINOv3ViTRopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self,
                hidden_size: int, 
                num_attention_heads: int, 
                image_size: int, 
                patch_size: int,
                rope_theta: float = 100.0, 
                ):
        super().__init__()

        self.base = rope_theta
        self.head_dim = hidden_size // num_attention_heads
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        self.patch_size = patch_size

        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)  # (head_dim / 4,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # Although we could precompute static patch_coords from image_size and patch_size in the config,
            # the model was trained with random_scale, so it can process images of varying sizes.
            # Therefore, it's better to compute patch_coords dynamically (with lru_cache).
            patch_coords = get_patches_center_coordinates(
                num_patches_h, num_patches_w, dtype=torch.float32, device=device
            )
            if self.training:
                patch_coords = augment_patches_center_coordinates(
                    patch_coords,
                    shift=None,
                    jitter=None,
                    rescale=2.0,
                )

            # (height * width, 2, head_dim / 4) -> (height * width, head_dim / 2) -> (height * width, head_dim)
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class DINOv3ViTAttention(nn.Module):
    """
    Multi-headed attention compatible with ALL_ATTENTION_FUNCTIONS.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float, key_bias: bool = True, value_bias: bool = True, query_bias: bool = True, proj_bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=key_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=value_bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=query_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=proj_bias)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, patches, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output

class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, hidden_size: int, layerscale_value: float) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(layerscale_value * torch.ones(hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


class DINOv3ViTMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, mlp_bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DINOv3ViTLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, hidden_size: int,
                    intermediate_size: Optional[int],
                    num_heads: int = 16, 
                    dropout: float = 0.0, 
                    key_bias: bool = False, 
                    value_bias: bool = True, 
                    query_bias: bool = True, 
                    proj_bias: bool = True, 
                    layerscale_value: float = 1.0, 
                    mlp_bias: bool = True, 
                    use_gated_mlp: bool = False, 
                    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = DINOv3ViTAttention(hidden_size, num_heads, dropout, key_bias, value_bias, query_bias, proj_bias)
        self.layer_scale1 = DINOv3ViTLayerScale(hidden_size, layerscale_value)

        self.norm2 = nn.LayerNorm(hidden_size)

        intermediate_size = 4 * hidden_size if intermediate_size is None else intermediate_size

        self.mlp = DINOv3ViTMLP(hidden_size, intermediate_size, mlp_bias)
        self.layer_scale2 = DINOv3ViTLayerScale(hidden_size, layerscale_value)
