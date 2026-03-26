# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import warnings

SUPPRESSED_WARNINGS: list[str] = [
    r".*Please use the new API settings to control TF32 behavior.*",
    r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
    r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
    r"^Grad strides do not match bucket view strides.*",
    r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
    r".*functools.partial will be a method descriptor in future Python versions*",
    r".*barrier\(\): using the device under current context.*",
]


def suppress_warnings() -> None:
    for pattern in SUPPRESSED_WARNINGS:
        warnings.filterwarnings("ignore", message=pattern)
