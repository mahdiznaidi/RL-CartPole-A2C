"""
A2C algorithm components
"""

from .buffer import RolloutBuffer
from .returns import compute_returns_1step, compute_returns_nstep
from .losses import compute_actor_loss, compute_critic_loss, compute_entropy
from .a2c_agent import A2CAgent

__all__ = [
    'RolloutBuffer',
    'compute_returns_1step',
    'compute_returns_nstep',
    'compute_actor_loss',
    'compute_critic_loss',
    'compute_entropy',
    'A2CAgent',
]