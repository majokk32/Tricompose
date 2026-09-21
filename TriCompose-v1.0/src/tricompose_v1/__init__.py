"""TriCompose version-1 inference-time composition interfaces."""

from .graph import CandidateEdge, CandidateGraph, CandidateNode, SelectionPolicy
from .registry import ModelRegistry, ModelSpec

__all__ = [
    "CandidateEdge",
    "CandidateGraph",
    "CandidateNode",
    "ModelRegistry",
    "ModelSpec",
    "SelectionPolicy",
]
