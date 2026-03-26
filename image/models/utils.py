# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import logging
import math
import os
from types import MethodType
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers.pytorch_utils import compile_compatible_method_lru_cache


# ---------------------------------------------------------------------------
# Patch coordinate utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Rotary position embedding utilities
# ---------------------------------------------------------------------------

def rotate_half(x: Tensor) -> Tensor:
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

    cos = cos.to(q_patches.dtype)
    sin = sin.to(q_patches.dtype)

    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)

    return q, k


# ---------------------------------------------------------------------------
# Fused QKV projection utilities
# ---------------------------------------------------------------------------

def _fused_forward_local(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Fused QKV forward for local DINOv3ViTAttention."""
    batch_size, patches, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

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

    Call AFTER model construction and checkpoint loading.
    Returns the number of attention modules fused.
    """
    count = 0
    for _, child in module.named_modules():
        if hasattr(child, 'q_proj') and hasattr(child, 'k_proj') and hasattr(child, 'v_proj') and hasattr(child, 'o_proj'):
            _fuse_attention_qkv(child)
            count += 1
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logging.info(f"Fused QKV projections in {count} attention modules")
    return count
