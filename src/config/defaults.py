"""
Default hyperparameters and configuration
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DefaultConfig:
    """
    Default configuration values shared across all agents
    """
    
    # Environment
    env_name: str = "CartPole-v1"
    
    # Network architecture
    state_dim: int = 4
    action_dim: int = 2
    hidden_size: int = 64
    activation: str = "tanh"  # tanh or relu
    
    # Training hyperparameters
    gamma: float = 0.99  # Discount factor
    actor_lr: float = 1e-5  # Actor learning rate
    critic_lr: float = 1e-3  # Critic learning rate
    
    # Training budget
    max_steps: int = 500_000  # Total training steps
    
    # Logging and evaluation
    log_interval: int = 1_000  # Log training metrics every N steps
    eval_interval: int = 20_000  # Evaluate every N steps
    eval_episodes: int = 10  # Number of episodes for evaluation
    
    # Device
    device: str = "cuda"  # cuda or cpu
    
    # Output directories
    output_dir: str = "outputs"
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.gamma > 0 and self.gamma <= 1, "gamma must be in (0, 1]"
        assert self.actor_lr > 0, "actor_lr must be positive"
        assert self.critic_lr > 0, "critic_lr must be positive"
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.activation in ["tanh", "relu"], "activation must be tanh or relu"
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'env_name': self.env_name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_size': self.hidden_size,
            'activation': self.activation,
            'gamma': self.gamma,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'max_steps': self.max_steps,
            'log_interval': self.log_interval,
            'eval_interval': self.eval_interval,
            'eval_episodes': self.eval_episodes,
            'device': self.device,
            'output_dir': self.output_dir,
        }


if __name__ == "__main__":
    # Test default config
    print("Testing DefaultConfig...")
    
    config = DefaultConfig()
    print(f"Environment: {config.env_name}")
    print(f"Gamma: {config.gamma}")
    print(f"Actor LR: {config.actor_lr}")
    print(f"Critic LR: {config.critic_lr}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Max steps: {config.max_steps:,}")
    
    # Test to_dict
    config_dict = config.to_dict()
    print(f"\nConfig as dict: {len(config_dict)} keys")
    
    print("✓ DefaultConfig works!")