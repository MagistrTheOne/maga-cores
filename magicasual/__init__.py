"""Public educational building blocks."""
from .attention import CausalGQA
from .config import ModelConfig, load_config
from .layers import RMSNorm, SwiGLU, silu
from .model import MagiCasualLLM, TransformerBlock
from .rope import apply_rope, precompute_rope
__all__ = ["CausalGQA", "ModelConfig", "load_config", "RMSNorm", "SwiGLU", "silu",
           "MagiCasualLLM", "TransformerBlock", "apply_rope", "precompute_rope"]
