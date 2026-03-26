# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from models.layers import DINOv3ViTRopePositionEmbedding
from models.pmd import PlainMaskDecoder


class PMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        num_q: int = 100,
        num_blocks: int = 6,
        masked_attn_enabled: bool = True,
        lateral_bn: bool = True,
        interaction_indices: List[int] | torch.Tensor = [5, 11, 17, 23],
        lateral_projection: str = "mlp",
        residual_projection: bool = True,
        hidden_dim: Optional[int] = None,
        fuse_decoder_qkv: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.fuse_decoder_qkv = fuse_decoder_qkv
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.lateral_bn = lateral_bn
        self.interaction_indices = interaction_indices
        self.residual_projection = residual_projection

        if self.encoder.is_dinov3:
            self.num_heads = self.encoder.backbone.blocks[0].attention.num_heads
        else:
            self.num_heads = self.encoder.backbone.blocks[0].attn.num_heads

        if self.lateral_bn:
            self.bn = nn.ModuleList([
                nn.SyncBatchNorm(self.encoder.backbone.embed_dim)
                for _ in range(len(self.interaction_indices))
            ])

        if not self.encoder.is_dinov3:
            self.rope = DINOv3ViTRopePositionEmbedding(
                hidden_size = self.encoder.backbone.embed_dim,
                num_attention_heads = self.num_heads,
                image_size = self.encoder.image_size[0],
                patch_size = self.encoder.backbone.patch_embed.patch_size[0],
            )

        self.decoder = PlainMaskDecoder(
            embed_dim = self.encoder.backbone.embed_dim,
            hidden_dim = hidden_dim,
            num_prefix_tokens = self.encoder.backbone.num_prefix_tokens,
            grid_size = self.encoder.backbone.patch_embed.grid_size,
            patch_size = self.encoder.backbone.patch_embed.patch_size,
            num_classes = num_classes,
            num_q = num_q,
            num_blocks = num_blocks,
            masked_attn_enabled = masked_attn_enabled,
            interaction_indices = interaction_indices,
            lateral_projection = lateral_projection,
            residual_projection = residual_projection,
            num_heads = self.num_heads,
        )

    @property
    def attn_mask_probs(self) -> torch.Tensor:
        return self.decoder.attn_mask_probs

    def normalize_features(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        normed_features = []
        for i, feature in enumerate(features):
            if self.lateral_bn:
                feature = self.bn[i](feature.permute(0, 2, 1)).permute(0, 2, 1)
            normed_features.append(feature)
        return normed_features

    def dinov2_forward(
        self,
        x: torch.Tensor,
        encoder_frozen: bool = False,
    ) -> List[torch.Tensor]:
        backbone = self.encoder.backbone

        def _forward_features(x: torch.Tensor) -> List[torch.Tensor]:
            features: List[torch.Tensor] = []

            x = backbone.patch_embed(x)
            x = backbone._pos_embed(x)
            x = backbone.patch_drop(x)
            x = backbone.norm_pre(x)
            
            for idx, block in enumerate(backbone.blocks):
                x = block(x)
                if idx in self.interaction_indices:
                    features.append(backbone.norm(x))

            return features

        if encoder_frozen:
            with torch.no_grad():
                features = _forward_features(x) # there is an internal compile bug with inference_mode, so we use no_grad instead
        else:
            features = _forward_features(x)

        normed_features = self.normalize_features(features)

        return normed_features
    
    def dinov3_forward(
        self,
        x: torch.Tensor,
        encoder_frozen: bool = False,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass for DINOv3 models with optional frozen encoder."""

        def _forward_features(x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
            features: List[torch.Tensor] = []
            rope = self.encoder.backbone.rope_embeddings(x)
            x = self.encoder.backbone.patch_embed(x)
            
            for i, layer_module in enumerate(self.encoder.backbone.blocks):
                x = layer_module(
                    x,
                    attention_mask=None,
                    position_embeddings=rope,
                )

                if i in self.interaction_indices:
                    features.append(self.encoder.backbone.norm(x))
            return features, rope
        
        if encoder_frozen:
            with torch.no_grad():
                features, rope = _forward_features(x)
        else:
            features, rope = _forward_features(x)

        normed_features = self.normalize_features(features)

        return normed_features, rope

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        # ========== ENCODER FORWARD ==========

        if self.encoder.is_dinov3:
            outputs, rope = self.dinov3_forward(x, encoder_frozen=self.encoder.is_frozen)
        else:
            outputs = self.dinov2_forward(x, encoder_frozen=self.encoder.is_frozen)
            rope = self.rope(x)

        # ========== DECODER FORWARD ==========

        mask_logits_per_layer, class_logits_per_layer = self.decoder(outputs, rope)

        return mask_logits_per_layer, class_logits_per_layer
