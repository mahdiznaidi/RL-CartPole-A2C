"""
Agent-specific configurations
"""

from dataclasses import dataclass
from typing import Optional
from .defaults import DefaultConfig


@dataclass
class AgentConfig(DefaultConfig):
    """
    Configuration for a specific agent
    Extends DefaultConfig with agent-specific parameters
    """
    
    # Agent identification
    agent_id: int = 0
    seed: int = 42
    
    # A2C specific parameters
    num_workers: int = 1  # K: number of parallel environments
    n_steps: int = 1  # n: number of steps before update
    
    # Stochastic rewards (for Agent 1+)
    reward_mask_prob: float = 0.0  # Probability of masking reward (0.0 = no masking)
    
    # Experiment name
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        """Set agent-specific configurations automatically"""
        super().__post_init__()
        
        # Configure based on agent_id
        if self.agent_id == 0:
            # Agent 0: Basic A2C (K=1, n=1)
            self.num_workers = 1
            self.n_steps = 1
            self.reward_mask_prob = 0.0
            
        elif self.agent_id == 1:
            # Agent 1: + Stochastic rewards (90% masking)
            self.num_workers = 1
            self.n_steps = 1
            self.reward_mask_prob = 0.9
            
        elif self.agent_id == 2:
            # Agent 2: K=6 parallel workers
            self.num_workers = 6
            self.n_steps = 1
            self.reward_mask_prob = 0.0
            
        elif self.agent_id == 3:
            # Agent 3: n=6 step returns
            self.num_workers = 1
            self.n_steps = 6
            self.reward_mask_prob = 0.0
            
        elif self.agent_id == 4:
            # Agent 4: K=6 x n=6 combined
            self.num_workers = 6
            self.n_steps = 6
            self.reward_mask_prob = 0.0
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"agent{self.agent_id}_seed{self.seed}"
        
        # Validate
        assert self.num_workers > 0, "num_workers must be positive"
        assert self.n_steps > 0, "n_steps must be positive"
        assert 0 <= self.reward_mask_prob < 1, "reward_mask_prob must be in [0, 1)"
    
    def get_run_dir(self) -> str:
        """Get directory path for this run"""
        return f"{self.output_dir}/runs/agent{self.agent_id}/seed{self.seed}"
    
    def to_dict(self):
        """Convert config to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            'agent_id': self.agent_id,
            'seed': self.seed,
            'num_workers': self.num_workers,
            'n_steps': self.n_steps,
            'reward_mask_prob': self.reward_mask_prob,
            'experiment_name': self.experiment_name,
        })
        return base_dict


def get_agent_config(agent_id: int, seed: int = 42) -> AgentConfig:
    """
    Get configuration for a specific agent
    
    Args:
        agent_id: Agent number (0-4)
        seed: Random seed
        
    Returns:
        AgentConfig: Configuration for the agent
        
    Example:
        >>> config = get_agent_config(0, seed=42)
        >>> print(config.num_workers, config.n_steps)
        1 1
    """
    return AgentConfig(agent_id=agent_id, seed=seed)


if __name__ == "__main__":
    # Test all agent configurations
    print("Testing AgentConfig for all agents:\n")
    
    for agent_id in range(5):
        config = get_agent_config(agent_id, seed=42)
        
        print(f"Agent {agent_id}:")
        print(f"  K (workers): {config.num_workers}")
        print(f"  n (steps): {config.n_steps}")
        print(f"  Reward masking: {config.reward_mask_prob:.1f}")
        print(f"  Run directory: {config.get_run_dir()}")
        print(f"  Experiment name: {config.experiment_name}")
        print()
    
    # Test to_dict
    config = get_agent_config(2, seed=123)
    config_dict = config.to_dict()
    print(f"Config as dict has {len(config_dict)} keys")
    print(f"Sample keys: {list(config_dict.keys())[:5]}")
    
    print("\n✓ All agent configs work!")
