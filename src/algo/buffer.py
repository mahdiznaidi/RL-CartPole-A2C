"""
Rollout buffer for storing experience
"""

from typing import List, Dict, Any
import torch
import numpy as np


class RolloutBuffer:
    """
    Buffer for storing rollout data during collection
    
    Stores: states, actions, rewards, next_states, dones, truncated, log_probs
    
    Example:
        >>> buffer = RolloutBuffer()
        >>> buffer.push(state, action, reward, next_state, done, truncated, log_prob)
        >>> data = buffer.get()
        >>> buffer.clear()
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.truncated = []
        self.log_probs = []
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        truncated: bool,
        log_prob: float
    ) -> None:
        """
        Add one transition to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag (True if episode truly ended)
            truncated: Truncation flag (True if episode cut off at max steps)
            log_prob: Log probability of the action
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.truncated.append(truncated)
        self.log_probs.append(log_prob)
    
    def get(self) -> Dict[str, Any]:
        """
        Get all data from buffer as a dictionary
        
        Returns:
            Dictionary with keys: states, actions, rewards, next_states, 
                                 dones, truncated, log_probs
            Each value is a list
        """
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_states': self.next_states,
            'dones': self.dones,
            'truncated': self.truncated,
            'log_probs': self.log_probs,
        }
    
    def clear(self) -> None:
        """Clear all data from buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.truncated = []
        self.log_probs = []
    
    def __len__(self) -> int:
        """Return number of transitions in buffer"""
        return len(self.states)


if __name__ == "__main__":
    # Test rollout buffer
    print("Testing RolloutBuffer...")
    
    buffer = RolloutBuffer()
    
    # Add some transitions
    for i in range(5):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = 1.0
        next_state = np.random.randn(4)
        done = (i == 4)  # Last one is terminal
        truncated = False
        log_prob = -0.5
        
        buffer.push(state, action, reward, next_state, done, truncated, log_prob)
    
    print(f"✓ Added {len(buffer)} transitions")
    
    # Get data
    data = buffer.get()
    print(f"✓ Retrieved data: {len(data['states'])} states")
    print(f"  Rewards: {data['rewards']}")
    print(f"  Dones: {data['dones']}")
    
    # Clear
    buffer.clear()
    assert len(buffer) == 0
    print(f"✓ Buffer cleared: {len(buffer)} transitions")
    
    print("\n✓ RolloutBuffer tests passed!")