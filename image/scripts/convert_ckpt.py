#!/usr/bin/env python3
"""
Convert a checkpoint (old or new, full or decoder-only) to a clean
decoder-only checkpoint compatible with the current codebase.

Handles:
  - Lightning checkpoints (extracts state_dict)
  - torch.compile ._orig_mod key mangling
  - Encoder backbone weight stripping
  - Obsolete key removal (old separate_bn, decoder PE, linear decoder fallback)

Usage:
  python scripts/convert_ckpt.py <ckpt_path> <output_path> [--config <yaml>] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ENCODER_PREFIXES = (
    "network.encoder.",
)

EXPECTED_PREFIXES = (
    "network.decoder.",
    "network.bn.",
    "network.rope.",
)

OBSOLETE_KEYS = (
    "criterion.empty_weight",
    "network._attn_mask_probs_fallback",
)

OBSOLETE_PREFIXES = (
    "network.decoder.pe.",       # old sin/learnable decoder PE
    "network.bn_prefix.",        # old separate_bn
    "network.bn_patch.",         # old separate_bn
)


def load_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def clean_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.replace("._orig_mod", ""): v for k, v in sd.items()}


def strip_encoder(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    out = {}
    n_stripped = 0
    for k, v in sd.items():
        if any(k.startswith(p) for p in ENCODER_PREFIXES):
            n_stripped += v.numel()
            continue
        out[k] = v
    return out, n_stripped


def drop_obsolete(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    dropped: list[str] = []
    out = {}
    for k, v in sd.items():
        if k in OBSOLETE_KEYS or any(k.startswith(p) for p in OBSOLETE_PREFIXES):
            dropped.append(k)
            continue
        out[k] = v
    return out, dropped


def validate_against_config(sd: dict[str, torch.Tensor], config_path: Path) -> None:
    """Optional: build model from config and check key compatibility."""
    try:
        import yaml
        from jsonargparse import ArgumentParser
    except ImportError:
        log.warning("jsonargparse/pyyaml not available, skipping config validation")
        return

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    network_cfg = cfg.get("model", {}).get("init_args", {}).get("network", {})
    if not network_cfg:
        log.warning("Could not find model.init_args.network in config, skipping validation")
        return

    log.info(f"Network class: {network_cfg.get('class_path', '?')}")
    log.info("(Full model instantiation skipped — would require encoder download)")

    model_keys = set(sd.keys())
    decoder_keys = {k for k in model_keys if k.startswith("network.decoder.")}
    bn_keys = {k for k in model_keys if k.startswith("network.bn.")}
    rope_keys = {k for k in model_keys if k.startswith("network.rope.")}
    other_keys = model_keys - decoder_keys - bn_keys - rope_keys

    log.info(f"  decoder keys: {len(decoder_keys)}")
    log.info(f"  bn keys:      {len(bn_keys)}")
    log.info(f"  rope keys:    {len(rope_keys)}")
    if other_keys:
        log.warning(f"  unexpected keys: {sorted(other_keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt_path", type=Path, help="Input checkpoint path")
    parser.add_argument("output_path", type=Path, help="Output checkpoint path")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML for optional validation")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without saving")
    args = parser.parse_args()

    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    log.info(f"Loading {args.ckpt_path}")
    sd = load_state_dict(args.ckpt_path)
    log.info(f"  {len(sd)} keys loaded")

    sd = clean_keys(sd)

    sd, n_encoder_params = strip_encoder(sd)
    if n_encoder_params:
        log.info(f"  Stripped {n_encoder_params:,} encoder params")

    sd, dropped = drop_obsolete(sd)
    if dropped:
        log.info(f"  Dropped {len(dropped)} obsolete keys: {dropped}")

    unexpected = [k for k in sd if not any(k.startswith(p) for p in EXPECTED_PREFIXES)]
    if unexpected:
        log.warning(f"  WARNING: {len(unexpected)} unexpected non-encoder keys: {unexpected}")

    log.info(f"  {len(sd)} keys remaining ({sum(v.numel() for v in sd.values()):,} params)")

    if args.config:
        validate_against_config(sd, args.config)

    if args.dry_run:
        log.info("Dry run — not saving. Final keys:")
        for k in sorted(sd.keys()):
            log.info(f"    {k}  {tuple(sd[k].shape)}")
        return

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd}, args.output_path)
    log.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
