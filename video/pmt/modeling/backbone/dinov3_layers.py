import logging
from types import MethodType
from typing import Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import compile_compatible_method_lru_cache

@compile_compatible_method_lru_cache(maxsize=32)
def get_patches_center_coordinates(
    num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Computes the 2D coordinates of the centers of image patches, normalized to the range [-1, +1].
    The center of each patch is exactly halfway between its top-left and bottom-right corners.

    Args:
        num_patches_h (int): Number of patches along the vertical (height) axis.
        num_patches_w (int): Number of patches along the horizontal (width) axis.
        dtype (torch.dtype): The desired data type of the returned tensor.

    Returns:
        torch.Tensor: A tensor of shape (height * width, 2), where each row contains the (y, x)
            coordinates of a patch center, normalized to [-1, +1].
    """
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    # (height, width, 2) -> (height * width, 2)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    return coords


def augment_patches_center_coordinates(
    coords: torch.Tensor,
    shift: Optional[float] = None,
    jitter: Optional[float] = None,
    rescale: Optional[float] = None,
) -> torch.Tensor:
    # Shift coords by adding a uniform value in [-shift, shift]
    if shift is not None:
        shift_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        shift_hw = shift_hw.uniform_(-shift, shift)
        coords = coords + shift_hw

    # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
    if jitter is not None:
        jitter_range = np.log(jitter)
        jitter_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        jitter_hw = jitter_hw.uniform_(-jitter_range, jitter_range).exp()
        coords = coords * jitter_hw

    # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
    if rescale is not None:
        rescale_range = np.log(rescale)
        rescale_hw = torch.empty(1, device=coords.device, dtype=coords.dtype)
        rescale_hw = rescale_hw.uniform_(-rescale_range, rescale_range).exp()
        coords = coords * rescale_hw

    return coords


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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    
def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors, but only to the patch tokens,
    ignoring the prefix tokens (cls token and register tokens).

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches  # cls token + register tokens

    q_prefix_tokens, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix_tokens, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    # apply rope only to patch tokens
    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)

    return q, k

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
        position_embeddings_type: str = 'rope',
        
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        if position_embeddings is not None and position_embeddings_type in ['sin', 'learnable']:
            index = patches - position_embeddings.shape[1]
            non_features, features = hidden_states[:, :index, :], hidden_states[:, index:, :]
            features = features + position_embeddings
            query_states = self.q_proj(torch.cat((non_features, features), dim=1))
            key_states = self.k_proj(torch.cat((non_features, features), dim=1))
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None and position_embeddings_type == 'rope':
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


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class DINOv3ViTDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


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
                    drop_path_rate: float = 0.0
                    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = DINOv3ViTAttention(hidden_size, num_heads, dropout, key_bias, value_bias, query_bias, proj_bias)
        self.layer_scale1 = DINOv3ViTLayerScale(hidden_size, layerscale_value)
        self.drop_path = DINOv3ViTDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(hidden_size)

        intermediate_size = 4 * hidden_size if intermediate_size is None else intermediate_size

        self.mlp = DINOv3ViTMLP(hidden_size, intermediate_size, mlp_bias)
        self.layer_scale2 = DINOv3ViTLayerScale(hidden_size, layerscale_value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


# ---------------------------------------------------------------------------
# Fused QKV projection utilities
# ---------------------------------------------------------------------------

def _fused_forward_local(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    position_embeddings_type: str = 'rope',
) -> torch.Tensor:
    """Fused QKV forward for local DINOv3ViTAttention."""
    batch_size, patches, _ = hidden_states.size()

    if position_embeddings is not None and position_embeddings_type in ('sin', 'learnable'):
        raise RuntimeError(
            "fused_qkv is incompatible with sin/learnable position embeddings. "
            "Use decoder_pe='rope' or disable fused_qkv."
        )

    qkv = self.qkv_proj(hidden_states)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is not None and position_embeddings_type == 'rope':
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, patches, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output


def _fused_forward_hf(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Fused QKV forward for HuggingFace DINOv3ViTAttention."""
    batch_size, patches, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, patches, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _fuse_attention_qkv(attn: nn.Module) -> None:
    """Fuse separate q/k/v projections into a single qkv_proj on a DINOv3ViTAttention module."""
    q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
    D = q_proj.weight.shape[0]
    device = q_proj.weight.device
    dtype = q_proj.weight.dtype

    qkv_weight = torch.cat([q_proj.weight.data, k_proj.weight.data, v_proj.weight.data], dim=0)

    q_bias = q_proj.bias.data if q_proj.bias is not None else torch.zeros(D, device=device, dtype=dtype)
    k_bias = k_proj.bias.data if k_proj.bias is not None else torch.zeros(D, device=device, dtype=dtype)
    v_bias = v_proj.bias.data if v_proj.bias is not None else torch.zeros(D, device=device, dtype=dtype)
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

    qkv_proj = nn.Linear(D, 3 * D, bias=True, device=device, dtype=dtype)
    qkv_proj.weight.data.copy_(qkv_weight)
    qkv_proj.bias.data.copy_(qkv_bias)

    attn.qkv_proj = qkv_proj
    del attn.q_proj, attn.k_proj, attn.v_proj

    is_hf = hasattr(attn, 'config')
    attn.forward = MethodType(_fused_forward_hf if is_hf else _fused_forward_local, attn)


def fuse_dinov3_qkv_projections(module: nn.Module) -> int:
    """Fuse separate q/k/v into single qkv_proj on all DINOv3ViTAttention modules.

    Call AFTER model construction and checkpoint loading, only during inference.
    Returns the number of attention modules fused.
    """
    print("Fusing QKV projections in DINOv3ViTAttention modules...")

    count = 0
    for _, child in module.named_modules():
        if hasattr(child, 'q_proj') and hasattr(child, 'k_proj') and hasattr(child, 'v_proj') and hasattr(child, 'o_proj'):
            _fuse_attention_qkv(child)
            count += 1
    logging.info(f"Fused QKV projections in {count} attention modules")
    return count

