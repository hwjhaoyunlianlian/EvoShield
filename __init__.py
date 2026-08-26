"""EvoShield reference implementation."""

from .defense import EvoShieldDefense
from .encoder import HashingEncoder
from .router import ConsensusRouter

__all__ = ["ConsensusRouter", "EvoShieldDefense", "HashingEncoder"]
__version__ = "0.1.0"
