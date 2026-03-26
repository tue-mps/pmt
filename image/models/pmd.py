# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.layers import ScaleBlock, DINOv3ViTLayer

class PlainMaskDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int], 
        num_prefix_tokens: int,
        grid_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        num_classes: int,
        num_q: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        interaction_indices: List[int] | torch.Tensor = [5, 11, 17, 23],
        lateral_projection: str = "mlp",
        residual_projection: bool = True,
        num_heads: int = 16,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.interaction_indices = interaction_indices
        self.residual_projection = residual_projection

        self.num_levels = len(self.interaction_indices)

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        ##### LATERAL PROJECTIONS #####
        if self.num_levels:
            if lateral_projection == 'linear':
                self.lateral_projections = nn.ModuleList([
                    nn.Linear(embed_dim, embed_dim)
                    for _ in range(self.num_levels)])
            elif lateral_projection == 'mlp':
                self.lateral_projections = nn.ModuleList([
                    nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.GELU(),
                    nn.Linear(embed_dim // 2, embed_dim))
                    for _ in range(self.num_levels)])
            else:
                self.lateral_projections = nn.ModuleList([
                    nn.Identity() 
                    for _ in range(self.num_levels)])

            self.scalers = nn.ParameterList([
                nn.Parameter(torch.ones(embed_dim))
                for _ in range(self.num_levels)])

        ##### DECODER #####
        self.q = nn.Embedding(num_q, embed_dim)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            decoder_block = DINOv3ViTLayer(embed_dim, intermediate_size=hidden_dim, num_heads=num_heads)
            self.blocks.append(decoder_block)

        self.decoder_norm = nn.LayerNorm(embed_dim)

        ##### HEADS #####
        max_patch_size = max(self.patch_size[0], self.patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(embed_dim) for _ in range(num_upscale)],
        )

        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.class_head = nn.Linear(embed_dim, num_classes + 1)

    def _predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask: torch.Tensor, prob: float) -> torch.Tensor:
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        pe: Optional[torch.Tensor],
    ) -> torch.Tensor:

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
        return module(x, mask, pe)

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int) -> torch.Tensor:
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(
            mask_logits,
            self.grid_size,
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        
        attn_mask[
            :,
            : self.num_q,
            self.num_q + self.num_prefix_tokens :,
        ] = (
            interpolated > 0
        )
        
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[i],
        )
            
        return attn_mask

    def block_forward(
        self,
        x: torch.Tensor,
        block: nn.Module,
        attn_mask: Optional[torch.Tensor],
        pe: Optional[torch.Tensor],
    ) -> torch.Tensor:
        
        attn_out = self._attn(block.attention, block.norm1(x), attn_mask, pe=pe)
        x = x + block.layer_scale1(attn_out)

        mlp_out = block.mlp(block.norm2(x))
        x = x + block.layer_scale2(mlp_out)
        return x
    
    def _lateral_projections_forward(self, lateral_features: List[torch.Tensor]) -> List[torch.Tensor]:
        for i, lateral_feature in enumerate(lateral_features):
            if self.residual_projection:
                lateral_feature = lateral_feature + self.lateral_projections[i](lateral_feature)
            else:
                lateral_feature = self.lateral_projections[i](lateral_feature)
            lateral_feature = lateral_feature * self.scalers[i]
            lateral_features[i] = lateral_feature

        return lateral_features

    def forward(
        self,
        outputs: List[torch.Tensor],
        pe: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        lateral_features = self._lateral_projections_forward(outputs)

        x = torch.stack(lateral_features, dim=0).sum(dim=0)

        x = torch.cat(
            (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
        )

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.blocks):
            if self.masked_attn_enabled:
                mask_logits, class_logits = self._predict(self.decoder_norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)
                
            x = self.block_forward(x, block, attn_mask, pe)

        mask_logits, class_logits = self._predict(self.decoder_norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
