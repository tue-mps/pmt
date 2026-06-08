# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the EoMT repository https://github.com/tue-mps/eomt/tree/master/models
# Used under the MIT License.
# ---------------------------------------------------------------
import logging
import math
import einops
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import Backbone, BACKBONE_REGISTRY

from .vit import ViT
from .scale_block import ScaleBlock

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from .dinov3_layers import DINOv3ViTRopePositionEmbedding, fuse_dinov3_qkv_projections
from ..decoders.video_decoder_online import Decoder as DecoderOnline 



class PMT_CLASS(nn.Module):
    def __init__(
        self,
        img_size,
        num_classes,
        name,
        num_frames,
        num_q: int = 200,
        num_blocks: int = 6,
        masked_attn_enabled: bool = True,
        extra_bn: bool = True,
        interaction_indices: List[int] | torch.Tensor = [5, 11, 17, 23],
        lateral_projection: str = "mlp",
        residual_projection: bool = True,
        residual_path: bool = True,
        level_scalers: bool = True, 
        decoder_pe: str = 'rope',
        hidden_dim: Optional[int] = None,
        decoder_type: str = 'online',
        frozen_encoder: bool = True,
        fused_qkv: bool = False,
    ):
        super().__init__()

        self.fused_qkv = fused_qkv

        if self.fused_qkv and decoder_pe != 'rope':
            raise ValueError("fused_qkv requires decoder_pe='rope'.")

        self.encoder = ViT(img_size=img_size, backbone_name=name, frozen_encoder=frozen_encoder)
        self.encoder.backbone.patch_embed.strict_img_size = False
        self.encoder.backbone.patch_embed.dynamic_img_pad = False
        self.encoder.backbone.dynamic_img_size = True
        self.num_q = num_q
        self.num_frames = num_frames
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.extra_bn = extra_bn
        self.interaction_indices = interaction_indices
        self.residual_projection = residual_projection
        self.residual_path = residual_path
        self.level_scalers = level_scalers
        self.decoder_pe = decoder_pe
        self.num_levels = len(self.interaction_indices) - 1
        self.decoder_type = decoder_type
       

        
        if self.encoder.is_dinov3:
            self.num_heads = self.encoder.backbone.blocks[0].attention.num_heads
        else:
            self.num_heads = self.encoder.backbone.blocks[0].attn.num_heads

        if self.extra_bn:
            self.bn = nn.ModuleList([nn.SyncBatchNorm(self.encoder.backbone.embed_dim)
             for _ in range(len(self.interaction_indices)) ])

        if not self.encoder.is_dinov3 and self.decoder_pe == 'rope':
            self.rope = DINOv3ViTRopePositionEmbedding(
                hidden_size = self.encoder.backbone.embed_dim,
                num_attention_heads = self.num_heads,
                image_size = self.encoder.image_size[0],
                patch_size = self.encoder.backbone.patch_embed.patch_size[0],
            )
        
        if self.decoder_type == 'online':
            self.decoder = DecoderOnline(
                    embed_dim = self.encoder.backbone.embed_dim,
                    hidden_dim = hidden_dim, # MLP hidden dim
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
                    residual_path = residual_path,
                    level_scalers = level_scalers,
                    num_heads = self.num_heads,
                    decoder_pe = decoder_pe,
                    num_frames = num_frames,
                )
        

     
        
       

  
    def normalize_features(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        normed_features = []
        for i, feature in enumerate(features):
            if self.extra_bn:
                feature = self.bn[i](feature.permute(0, 2, 1)).permute(0, 2, 1)
            normed_features.append(feature)
        return normed_features

   
    def dinov2_forward(
        self,
        x: torch.Tensor,
        encoder_frozen: bool = False,
    ) -> List[torch.Tensor]:
        backbone = self.encoder.backbone

        def _forward_features(x: torch.Tensor) -> torch.Tensor:
            features: List[torch.Tensor] = []

            x = backbone.patch_embed(x)
            _, H, W ,_= x.shape
            x = backbone._pos_embed(x)
            x = backbone.patch_drop(x)
            x = backbone.norm_pre(x)
            
            for idx, block in enumerate(backbone.blocks):
                x = block(x)
                if idx in self.interaction_indices:
                    features.append(backbone.norm(x))

            return features, H, W

        if encoder_frozen:
            with torch.no_grad():
                features, H, W = _forward_features(x) # there is an internal compile bug with inference_mode, so we use no_grad instead
        else:
            features, H, W = _forward_features(x)

        normed_features = self.normalize_features(features)

        return normed_features, H, W
    
    def dinov3_forward(
        self,
        x: torch.Tensor,
        encoder_frozen: bool = False,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass for DINOv3 models with optional frozen encoder."""

        def _forward_features(x: torch.Tensor) -> torch.Tensor:
            features: List[torch.Tensor] = []
            H , W = x.shape[2] // self.encoder.backbone.patch_embed.patch_size[0], x.shape[3] // self.encoder.backbone.patch_embed.patch_size[1]
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
            return features, rope, H, W
        
        if encoder_frozen:
            with torch.no_grad():
                features, rope, H, W = _forward_features(x)
        else:
            features, rope, H, W = _forward_features(x)

        normed_features = self.normalize_features(features)

        return normed_features, rope, H, W

    def eva_forward(
        self,
        x: torch.Tensor,
        encoder_frozen: bool = False,
    ) -> List[torch.Tensor]:

        def _forward_features(x: torch.Tensor) -> torch.Tensor:
            features: List[torch.Tensor] = []
            H , W = x.shape[2] // self.encoder.backbone.patch_embed.patch_size[0], x.shape[3] // self.encoder.backbone.patch_embed.patch_size[1]
            x = self.encoder.backbone.patch_embed(x)
            x, rope = self.encoder.backbone._pos_embed(x)
            x = self.encoder.backbone.norm_pre(x)

            for i, layer_module in enumerate(self.encoder.backbone.blocks):
                    x = layer_module(x, rope=rope)
                    if i in self.interaction_indices:
                        features.append(self.encoder.backbone.norm(x))

            return features, rope, H, W
        
        if encoder_frozen:
            with torch.no_grad():
                features, rope, H, W = _forward_features(x)
        else:
            features, rope, H, W = _forward_features(x)

        normed_features = self.normalize_features(features)

        return normed_features, rope, H, W

    def forward(
        self,
        x: torch.Tensor,
        resume: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        # ========== ENCODER FORWARD ==========

        if self.encoder.is_dinov3:
            outputs, rope, H, W = self.dinov3_forward(x, encoder_frozen=self.encoder.is_frozen)
        elif self.encoder.is_eva:
            outputs, _, H, W = self.eva_forward(x, encoder_frozen=self.encoder.is_frozen) # this should return rope
            # EVA rope is not compatible with DINOv3 layers in the decoder, recompute it for simplicity
            if self.decoder_pe == 'rope':
                rope = self.rope(x)
        else:
            outputs, H, W = self.dinov2_forward(x, encoder_frozen=self.encoder.is_frozen)

            if self.decoder_pe == 'rope':
                rope = self.rope(x)

        if self.decoder_pe != 'rope':
            rope = None

        # ========== DECODER FORWARD ==========

        if self.decoder_type == 'online':
            out = self.decoder(outputs, rope, H, W, resume=resume)
        else:
            out = self.decoder(outputs, rope, H, W)

        # Final predictions
        return out

    def _maybe_apply_fused_qkv(self) -> None:
        if not self.fused_qkv or not self.encoder.is_dinov3:
            return

        needs_fusion = any(
            hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj') and hasattr(module, 'o_proj')
            for _, module in self.named_modules()
        )
        if needs_fusion:
            print("Applying fused QKV projections for DINOv3...")
            fused_count = fuse_dinov3_qkv_projections(self)
            logging.info(f"Applied fused_qkv to {fused_count} attention modules")

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        try:
            incompatible = super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            incompatible = super().load_state_dict(state_dict, strict=strict)

        self._maybe_apply_fused_qkv()
        return incompatible

@BACKBONE_REGISTRY.register()
class PMT(PMT_CLASS, Backbone):
    def __init__(self, cfg, input_shape):
        self.img_size=cfg.MODEL.BACKBONE.IMG_SIZE
        self.num_q = cfg.MODEL.BACKBONE.NUM_OBJECT_QUERIES
        self.num_classes=cfg.MODEL.BACKBONE.NUM_CLASSES
        self.name=cfg.MODEL.BACKBONE.MODEL_NAME
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM  
        self.extra_bn = cfg.MODEL.BACKBONE.EXTRA_BN
        self.num_blocks = cfg.MODEL.BACKBONE.NUM_BLOCKS
        self.masked_attn_enabled = cfg.MODEL.BACKBONE.MASKED_ATTN_ENABLED
        self.interaction_indices = cfg.MODEL.BACKBONE.INTERACTION_INDICES
        self.residual_projection = cfg.MODEL.BACKBONE.RESIDUAL_PROJECTION
        self.residual_path = cfg.MODEL.BACKBONE.RESIDUAL_PATH
        self.level_scalers = cfg.MODEL.BACKBONE.LEVEL_SCALERS
        self.decoder_pe = cfg.MODEL.BACKBONE.DECODER_PE
        self.decoder_type = cfg.MODEL.BACKBONE.DECODER_TYPE
        self.frozen_encoder = cfg.MODEL.BACKBONE.FROZEN_ENCODER
        self.lateral_projection = cfg.MODEL.BACKBONE.LATERAL_PROJECTION
        self.start_steps = cfg.MODEL.BACKBONE.START_STEPS
        self.end_steps = cfg.MODEL.BACKBONE.END_STEPS
        self.hidden_dim = cfg.MODEL.BACKBONE.HIDDEN_DIM
        self.fused_qkv = cfg.MODEL.BACKBONE.FUSED_QKV
        
        
        
        super().__init__(
            img_size=self.img_size,
            num_classes=self.num_classes,
            name=self.name,
            num_q = self.num_q,
            num_frames= self.num_frames,
            extra_bn = self.extra_bn,
            num_blocks = self.num_blocks,
            masked_attn_enabled = self.masked_attn_enabled,
            interaction_indices = self.interaction_indices, 
            residual_projection = self.residual_projection,
            residual_path = self.residual_path,
            level_scalers = self.level_scalers,
            decoder_pe = self.decoder_pe,
            decoder_type = self.decoder_type,
            frozen_encoder = self.frozen_encoder,
            lateral_projection= self.lateral_projection,
            hidden_dim = self.hidden_dim,
            fused_qkv = self.fused_qkv,
            
        )

       