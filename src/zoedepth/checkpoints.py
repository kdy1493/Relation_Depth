"""Checkpoint utilities for ZoeDepth."""

from __future__ import annotations

from typing import Any

import torch


def unwrap_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict
