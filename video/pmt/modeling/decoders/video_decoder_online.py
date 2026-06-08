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
import einops

from timm.layers import trunc_normal_
from ..backbone.scale_block import ScaleBlock, Mask2FormerSinePositionEmbedding
from ..backbone.dinov3_layers import DINOv3ViTLayer

class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int], 
        num_prefix_tokens: int,
        grid_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        num_classes: int,
        num_q: int,
        num_blocks: int = 6,
        masked_attn_enabled: bool = True,
        interaction_indices: List[int] | torch.Tensor = [5, 11, 17, 23],
        lateral_projection: str = "mlp",
        residual_projection: bool = True,
        residual_path: bool = True, # Not used
        level_scalers: bool = True,
        num_heads: int = 16,
        decoder_pe: str = 'rope',
        num_frames: int = 5,
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
        self.level_scalers = level_scalers
        self.decoder_pe = decoder_pe
        self.mask_classification = True
        self.num_frames = num_frames

        if self.decoder_pe == 'sin':
            self.pe = Mask2FormerSinePositionEmbedding(
                num_pos_feats=embed_dim // 2, normalize=True
            )
        elif self.decoder_pe == 'learnable':
            self.pe = nn.Parameter(torch.randn(1, grid_size[0] * grid_size[1], embed_dim))
            trunc_normal_(self.pe, std=.02)
            
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

            if self.level_scalers:
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

        ##### QUERY PROPAGATION #####
        self.query_updater = nn.Linear(embed_dim, embed_dim)
        self.last_query_embed = None

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

    def _predict(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, H, W
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits, q

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

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class, outputs_seg_masks)
            ]
        else:
            
            return [{"pred_masks": b} for b in outputs_seg_masks]

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        pe: Optional[torch.Tensor],
    ) -> torch.Tensor:

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
        return module(x, mask, pe, self.decoder_pe)

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int, H: int, W: int) -> torch.Tensor:
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(
            mask_logits,
            (H, W),
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        
        # Mask query-to-patch interactions
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

    def _clear_memory(self):
        del self.last_query_embed
        self.last_query_embed = None
        return


    def segmenter(self, x: torch.Tensor, blocks: nn.ModuleList, H: int, W: int, pe: Optional[torch.Tensor], resume: bool = False) -> dict:
        """Process video frames sequentially with query propagation across frames.
        
        Args:
            x: Input tensor of shape (bt, n, c) where bt = b*t
            blocks: Decoder blocks to process through
            H: Height of feature map
            W: Width of feature map
            pe: Positional encoding
            resume: Whether to resume from previous temporal state
        
        Returns:
            Dictionary with predictions and auxiliary outputs
        """
        attn_mask = None
        mask_logits_per_frame, class_logits_per_frame = [], []
        mask_logits_per_frame_list, class_logits_per_frame_list = [], []
        
        bt = x.shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        
        # Reshape to process frame by frame: (bt, n, c) -> (b, t, n, c)
        x = einops.rearrange(x, '(b t) n c -> b t n c', t=t)
        x_outputs = []

        for i in range(t):
            mask_logits_per_layer, class_logits_per_layer = [], []
            
            if i == 0 and resume is False:
                # First frame: initialize queries
                self._clear_memory()
                x_frame = x[:, i, :, :]  # (b, n, c)
                
                for j, block in enumerate(blocks):
                    if j == 0:
                        # Attach learnable queries at the beginning
                        x_frame = torch.cat(
                            (self.q.weight[None, :, :].expand(bs, -1, -1), x_frame), dim=1
                        )
                    
                    if self.training:
                        mask_logits, class_logits, _ = self._predict(x_frame, H, W)
                        mask_logits_per_layer.append(mask_logits)
                        class_logits_per_layer.append(class_logits)
                    
                    x_frame = self.block_forward(x_frame, block, attn_mask, pe)
                
                x_outputs.append(x_frame)
                # Propagate queries to next frame
                propag_query = self.query_updater(x_frame[:, :self.num_q, :])
                self.last_query_embed = propag_query + self.q.weight[None, :, :]
                
            else:
                # Subsequent frames: use propagated queries
                x_frame = x[:, i, :, :]  # (b, n, c)
                
                for j, block in enumerate(blocks):
                    if j == 0:
                        # Attach propagated queries
                        x_frame = torch.cat(
                            (self.last_query_embed, x_frame), dim=1
                        )
                    
                    if self.training:
                        mask_logits, class_logits, _ = self._predict(x_frame, H, W)
                        mask_logits_per_layer.append(mask_logits)
                        class_logits_per_layer.append(class_logits)
                    
                    x_frame = self.block_forward(x_frame, block, attn_mask, pe)
                
                x_outputs.append(x_frame)
                # Update queries for next frame
                propag_query = self.query_updater(x_frame[:, :self.num_q, :])
                self.last_query_embed = propag_query + self.q.weight[None, :, :]
            
            # Collect predictions per frame and layer
            if self.training:
                mask_logits_per_layer = torch.stack(mask_logits_per_layer, dim=0)  # (L, b, q, h, w)
                class_logits_per_layer = torch.stack(class_logits_per_layer, dim=0)  # (L, b, q, c)
                
                mask_logits_per_frame.append(mask_logits_per_layer)
                class_logits_per_frame.append(class_logits_per_layer)
        
        # Stack across time: (L, t, b, q, h, w) and (L, t, b, q, c)
        if self.training:
            mask_logits_per_frame = torch.stack(mask_logits_per_frame, dim=1)
            class_logits_per_frame = torch.stack(class_logits_per_frame, dim=1)
            
            # Convert to expected output format: (b, q, t, h, w) and (b, t, q, c)
            for i in range(len(mask_logits_per_frame)):
                mask_logits_per_frame_list.append(
                    mask_logits_per_frame[i].permute(1, 2, 0, 3, 4)  # (L, t, b, q, h, w) -> (b, q, t, h, w)
                )
            
            for i in range(len(class_logits_per_frame)):
                class_logits_per_frame_list.append(
                    class_logits_per_frame[i].permute(1, 0, 2, 3)  # (L, t, b, q, c) -> (b, t, q, c)
                )
        
        # Get final predictions
        x_outputs = torch.stack(x_outputs, dim=0)  # (t, b, n, c)
        x_outputs = x_outputs.reshape(t * bs, x_outputs.shape[2], x_outputs.shape[3])
        mask_logits, class_logits, _ = self._predict(self.decoder_norm(x_outputs), H, W)
        mask_logits = einops.rearrange(mask_logits, '(b t) q h w -> b q t h w', t=t)
        class_logits = einops.rearrange(class_logits, '(b t) q c -> b t q c', t=t)
        
        out = {
            'pred_logits': class_logits,
            'pred_masks': mask_logits,
            'aux_outputs': self._set_aux_loss(
                class_logits_per_frame_list if self.mask_classification else None,
                mask_logits_per_frame_list,
            ),
        }
        
        return out
    
    def _lateral_projections_forward(self, lateral_features: List[torch.Tensor]) -> List[torch.Tensor]:
        for i, lateral_feature in enumerate(lateral_features):
            if self.residual_projection:
                lateral_feature = lateral_feature + self.lateral_projections[i](lateral_feature)
            else:
                lateral_feature = self.lateral_projections[i](lateral_feature)
            lateral_features[i] = lateral_feature * self.scalers[i]

        return lateral_features

    def forward(
        self,
        outputs: List[torch.Tensor],
        pe: Optional[torch.Tensor],
        H: int,
        W: int,
        resume: bool = False,
    ) -> dict:
        """Forward pass for video decoder.
        
        Args:
            outputs: List of lateral features from backbone
            pe: Positional encoding
            H: Height of feature map
            W: Width of feature map
            resume: If True, resume from previous state
        
        Returns:
            Dictionary with predictions
        """
        lateral_features = self._lateral_projections_forward(outputs)

        # Sum all lateral features
        x = torch.stack(lateral_features, dim=0).sum(dim=0)

        if self.decoder_pe == 'sin':
            size = torch.Size((x.shape[0], self.embed_dim, self.grid_size[0], self.grid_size[1]))
            pe = self.pe(size, x.device, x.dtype).flatten(2).transpose(1, 2)
        elif self.decoder_pe == 'learnable':
            pe = self.pe
        
        return self.segmenter(x, self.blocks, H, W, pe, resume=resume)
        
        