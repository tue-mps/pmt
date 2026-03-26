# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import logging
import os
import torch
import torch.nn as nn

import timm
from transformers import AutoModel

class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int = 16,
        backbone_name: str = "vit_large_patch14_reg4_dinov2",
        frozen_encoder: bool = False,
    ):
        super().__init__()

        self.image_size = img_size

        self.is_dinov3 = False
        
        if "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(
                    backbone_name,
                    **self._get_transformers_offline_kwargs(),
                ),
                img_size,
            )
            self.is_dinov3 = True
        else:
            if self._offline_env_enabled():
                self._force_timm_hf_local_only()
                
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self.is_frozen = frozen_encoder
        if frozen_encoder:
            self.freeze_encoder()

    @staticmethod
    def _offline_env_enabled() -> bool:
        return bool(os.getenv("HF_HUB_OFFLINE") or os.getenv("TRANSFORMERS_OFFLINE"))

    @staticmethod
    def _force_timm_hf_local_only() -> None:
        if getattr(timm, "_hf_hub_local_only_patched", False):
            return
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return

        try:
            import timm.models._hub as timm_hub
        except Exception:
            return

        def _hf_hub_download_local_only(*args, **kwargs):
            kwargs.setdefault("local_files_only", True)
            # Always set cache_dir from HF_HOME when offline mode is enabled
            # This ensures consistency with where weights were downloaded
            hf_home = os.getenv("HF_HOME")
            if hf_home:
                # Always override cache_dir with HF_HOME to ensure consistency
                original_cache_dir = kwargs.get("cache_dir")
                kwargs["cache_dir"] = hf_home
                # Verify cache directory exists
                if not os.path.exists(hf_home):
                    error_msg = f"HF_HOME cache directory does not exist: {hf_home}. Please ensure weights are downloaded first."
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        logging.error(error_msg)
                    raise FileNotFoundError(error_msg)
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    model_id = args[0] if args else kwargs.get("repo_id", "unknown")
                    filename = args[1] if len(args) > 1 else kwargs.get("filename", "unknown")
                    logging.info(f"Loading from HF_HOME cache_dir: {hf_home} (overriding {original_cache_dir}) for {model_id}/{filename}")
            else:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    logging.warning("HF_HOME not set, using default cache location")
            try:
                return hf_hub_download(*args, **kwargs)
            except Exception as e:
                # Provide helpful error message if files not found
                if "LocalEntryNotFoundError" in str(type(e).__name__) or "Cannot find" in str(e):
                    model_id = args[0] if args else kwargs.get("repo_id", "unknown")
                    filename = args[1] if len(args) > 1 else kwargs.get("filename", "unknown")
                    cache_dir_used = kwargs.get("cache_dir", "default")
                    error_msg = (
                        f"Model weights not found in cache. "
                        f"Model: {model_id}, File: {filename}, Cache: {cache_dir_used}. "
                        f"Please run scripts/download_weights.py with HF_HOME={cache_dir_used} set."
                    )
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        logging.error(error_msg)
                    raise FileNotFoundError(error_msg) from e
                raise

        timm_hub.hf_hub_download = _hf_hub_download_local_only
        timm._hf_hub_local_only_patched = True

    @staticmethod
    def _get_transformers_offline_kwargs() -> dict:
        """Get kwargs for transformers AutoModel.from_pretrained() when offline mode is enabled."""
        kwargs = {}
        if ViT._offline_env_enabled():
            kwargs["local_files_only"] = True
            # Set cache_dir from HF_HOME if available
            hf_home = os.getenv("HF_HOME")
            if hf_home:
                kwargs["cache_dir"] = hf_home
        return kwargs

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        backbone.patch_embed = backbone.embeddings
        backbone.patch_embed.patch_size = (
            backbone.embeddings.config.patch_size,
            backbone.embeddings.config.patch_size,
        )
        backbone.patch_embed.grid_size = (
            img_size[0] // backbone.embeddings.config.patch_size,
            img_size[1] // backbone.embeddings.config.patch_size,
        )

        backbone.embed_dim = backbone.embeddings.config.hidden_size
        backbone.num_prefix_tokens = backbone.patch_embed.config.num_register_tokens + 1
        backbone.blocks = backbone.layer

        del (
            backbone.patch_embed.mask_token,
            backbone.embeddings,
            backbone.layer,
        )

        return backbone

    def freeze_encoder(self):
        """Freeze the backbone encoder parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logging.info("Backbone encoder frozen.")
