"""Flood-depth estimation from images of partially submerged vehicles."""

from .pipeline import InferenceConfig, run_inference

__all__ = ["InferenceConfig", "run_inference"]
