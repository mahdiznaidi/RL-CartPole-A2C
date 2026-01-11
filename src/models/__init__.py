"""
Neural network models for A2C
"""

from .actor import ActorNetwork
from .critic import CriticNetwork
from .init_weights import init_weights

__all__ = [
    'ActorNetwork',
    'CriticNetwork',
    'init_weights',
]