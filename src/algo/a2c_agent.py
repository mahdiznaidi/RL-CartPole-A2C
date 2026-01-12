"""
A2C Agent - Main agent class that combines everything
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional

from src.models import ActorNetwork, CriticNetwork
from src.utils import to_tensor, to_numpy
from .buffer import RolloutBuffer
from .returns import compute_returns_1step, compute_returns_nstep, compute_returns_nstep_grouped
from .losses import compute_actor_loss, compute_critic_loss, compute_entropy


class A2CAgent:
    """
    Advantage Actor-Critic Agent
    
    Combines actor (policy) and critic (value) networks with
    appropriate optimizers and update logic.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        config: Configuration object with hyperparameters
        
    Example:
        >>> from src.config import get_agent_config
        >>> config = get_agent_config(0, seed=42)
        >>> agent = A2CAgent(4, 2, config)
        >>> 
        >>> # Select action
        >>> state = np.random.randn(4)
        >>> action, log_prob = agent.select_action(state)
        >>> 
        >>> # Update
        >>> buffer = RolloutBuffer()
        >>> # ... collect data ...
        >>> next_states = [np.random.randn(4)]
        >>> metrics = agent.update(buffer, next_states)
    """
    
    def __init__(self, state_dim: int, action_dim: int, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Get device
        if torch.cuda.is_available() and config.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"A2C Agent using device: {self.device}")
        
        # Create networks
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=config.hidden_size,
            activation=config.activation
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_size=config.hidden_size,
            activation=config.activation
        ).to(self.device)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr
        )
        
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
    def select_action(
        self,
        state: np.ndarray,
        greedy: bool = False
    ) -> Tuple[int, float]:
        """
        Select action given state
        
        Args:
            state: Current state, shape (state_dim,)
            greedy: If True, select best action (argmax). If False, sample.
            
        Returns:
            action: Selected action (int)
            log_prob: Log probability of action (float)
            
        Example:
            >>> state = np.random.randn(4)
            >>> action, log_prob = agent.select_action(state, greedy=False)
        """
        # Convert to tensor
        state_tensor = to_tensor(state, self.device).unsqueeze(0)  # Add batch dim
        
        with torch.no_grad():
            # Get action distribution
            dist = self.actor.get_action_distribution(state_tensor)
            
            if greedy:
                # Select best action
                action_probs = self.actor(state_tensor)
                action = action_probs.argmax(dim=-1)
            else:
                # Sample from distribution
                action = dist.sample()
            
            # Get log probability
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def update(
        self,
        buffer: RolloutBuffer,
        final_next_states: list
    ) -> Dict[str, float]:
        """
        Update actor and critic based on collected experience
        
        Args:
            buffer: RolloutBuffer with collected transitions
            final_next_states: List of next states at end of each trajectory
                              (used for bootstrapping at trajectory boundaries)
            
        Returns:
            metrics: Dictionary with losses and other metrics
            
        Steps:
            1. Get data from buffer
            2. Compute state values V(s) and V(s')
            3. Compute returns (1-step or n-step based on config)
            4. Compute advantages
            5. Update critic
            6. Update actor
            7. Return metrics
        """
        # Get data from buffer
        data = buffer.get()
        n = len(data['states'])
        
        if n == 0:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
        
        # Convert to tensors
        states = torch.stack([to_tensor(s, self.device) for s in data['states']])
        actions = torch.tensor(data['actions'], device=self.device)
        rewards = data['rewards']
        next_states = torch.stack([to_tensor(s, self.device) for s in data['next_states']])
        log_probs = torch.tensor(data['log_probs'], device=self.device)
        
        # Compute values
        values = self.critic(states).squeeze(-1)  # (n,)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1)  # (n,)
        
        # Compute returns based on n_steps
        if self.config.n_steps == 1:
            targets = compute_returns_1step(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=data['dones'],
                truncated=data['truncated'],
                gamma=self.config.gamma
            ).to(self.device)
        else:
            if 'env_ids' in data and data['env_ids']:
                targets = compute_returns_nstep_grouped(
                    rewards=rewards,
                    values=values,
                    next_values=next_values,
                    dones=data['dones'],
                    truncated=data['truncated'],
                    gamma=self.config.gamma,
                    n_steps=self.config.n_steps,
                    env_ids=data['env_ids'],
                ).to(self.device)
            else:
                targets = compute_returns_nstep(
                    rewards=rewards,
                    values=values,
                    next_values=next_values,
                    dones=data['dones'],
                    truncated=data['truncated'],
                    gamma=self.config.gamma,
                    n_steps=self.config.n_steps
                ).to(self.device)
        
        # Compute advantages
        with torch.no_grad():
            advantages = targets - values
            # Normalize advantages (optional but helps stability)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update critic
        critic_loss = compute_critic_loss(values, targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping (optional but recommended)
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        # Update actor
        # Need to recompute action probs for current policy
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        actor_loss = compute_actor_loss(new_log_probs, advantages.detach())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Compute entropy for logging
        with torch.no_grad():
            entropy = compute_entropy(action_probs)
        
        # Return metrics
        metrics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'mean_value': values.mean().item(),
            'mean_advantage': advantages.mean().item(),
        }
        
        return metrics
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for a state
        
        Args:
            state: State array
            
        Returns:
            value: Estimated value V(s)
        """
        state_tensor = to_tensor(state, self.device).unsqueeze(0)
        
        with torch.no_grad():
            value = self.critic(state_tensor).item()
        
        return value
    
    def save(self, path: str):
        """Save agent networks"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Saved agent to {path}")
    
    def load(self, path: str):
        """Load agent networks"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Loaded agent from {path}")


if __name__ == "__main__":
    # Test A2C agent
    print("Testing A2CAgent...")
    
    import sys
    sys.path.insert(0, '..')
    
    from src.config import get_agent_config
    
    # Create agent
    config = get_agent_config(0, seed=42)
    agent = A2CAgent(state_dim=4, action_dim=2, config=config)
    
    print("\n✓ Agent created")
    
    # Test action selection
    state = np.random.randn(4)
    action, log_prob = agent.select_action(state, greedy=False)
    print(f"✓ Selected action: {action}, log_prob: {log_prob:.4f}")
    
    greedy_action, _ = agent.select_action(state, greedy=True)
    print(f"✓ Greedy action: {greedy_action}")
    
    # Test update
    buffer = RolloutBuffer()
    for i in range(10):
        s = np.random.randn(4)
        a, lp = agent.select_action(s)
        r = 1.0
        s_next = np.random.randn(4)
        done = (i == 9)
        trunc = False
        
        buffer.push(s, a, r, s_next, done, trunc, lp)
    
    metrics = agent.update(buffer, [np.random.randn(4)])
    print(f"✓ Update works")
    print(f"  Actor loss: {metrics['actor_loss']:.4f}")
    print(f"  Critic loss: {metrics['critic_loss']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")
    
    # Test value estimation
    value = agent.get_value(state)
    print(f"✓ Value estimation: {value:.2f}")
    
    print("\n✓ A2CAgent tests passed!")
