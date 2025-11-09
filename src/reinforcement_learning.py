"""
Reinforcement Learning Module for CoT Generation

This module implements RL-based chain-of-thought generation including:
- Policy network for action selection
- Value network for state estimation
- Reward function for CoT quality
- PPO algorithm for stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque


class PolicyNetwork(nn.Module):
    """
    Policy network for selecting CoT actions.
    Outputs action probabilities for next reasoning step.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 100,
        hidden_dims: List[int] = [512, 256],
        use_lstm: bool = True,
        lstm_hidden_size: int = 256
    ):
        super().__init__()
        
        self.use_lstm = use_lstm
        
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=state_dim,
                hidden_size=lstm_hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
            state_dim = lstm_hidden_size
        
        # Policy head
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.policy_head = nn.Sequential(*layers)
        
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Args:
            state: [batch_size, seq_len, state_dim] or [batch_size, state_dim]
            hidden: LSTM hidden state (optional)
            
        Returns:
            action_logits: [batch_size, action_dim]
            hidden: Updated LSTM hidden state
        """
        if self.use_lstm:
            if state.dim() == 2:
                state = state.unsqueeze(1)  # Add sequence dimension
            
            lstm_out, hidden = self.lstm(state, hidden)
            state = lstm_out[:, -1, :]  # Take last timestep
        
        action_logits = self.policy_head(state)
        return action_logits, hidden


class ValueNetwork(nn.Module):
    """
    Value network for estimating state value.
    Used for advantage estimation in RL.
    """
    
    def __init__(
        self,
        state_dim: int = 768,
        hidden_dims: List[int] = [512, 256]
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Single value output
        self.value_head = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            
        Returns:
            value: [batch_size, 1]
        """
        return self.value_head(state)


class RewardFunction:
    """
    Multi-component reward function for CoT quality.
    """
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.components = config['reward']['components']
        
    def compute_diagnosis_reward(
        self,
        pred_diagnosis: torch.Tensor,
        true_diagnosis: torch.Tensor
    ) -> float:
        """Reward for correct diagnosis."""
        correct = (pred_diagnosis.argmax(dim=-1) == true_diagnosis).float()
        return correct.mean().item()
    
    def compute_attention_localization_reward(
        self,
        attention_map: torch.Tensor,
        saliency_map: torch.Tensor
    ) -> float:
        """Reward for attention aligned with saliency."""
        # Normalize both maps
        attention_norm = F.normalize(attention_map.flatten(1), p=1, dim=1)
        saliency_norm = F.normalize(saliency_map.flatten(1), p=1, dim=1)
        
        # Compute overlap (similarity)
        overlap = (attention_norm * saliency_norm).sum(dim=1).mean()
        return overlap.item()
    
    def compute_coherence_reward(
        self,
        reasoning_text: List[str],
        model=None
    ) -> float:
        """Reward for coherent reasoning (low perplexity)."""
        # Simplified: compute based on text length and repetition
        if not reasoning_text:
            return 0.0
        
        avg_length = np.mean([len(text.split()) for text in reasoning_text])
        unique_ratio = len(set(' '.join(reasoning_text).split())) / len(' '.join(reasoning_text).split())
        
        # Encourage moderate length and diversity
        length_score = 1.0 - abs(avg_length - 15) / 15  # Target ~15 words
        diversity_score = unique_ratio
        
        return (length_score + diversity_score) / 2
    
    def compute_region_relevance_reward(
        self,
        region_attention: torch.Tensor
    ) -> float:
        """Reward for focused attention (low entropy)."""
        # Compute entropy of attention distribution
        probs = F.softmax(region_attention, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Lower entropy = more focused = higher reward
        max_entropy = torch.log(torch.tensor(region_attention.shape[-1], dtype=torch.float))
        normalized_entropy = entropy / max_entropy
        
        return (1.0 - normalized_entropy).item()
    
    def compute_step_diversity_reward(
        self,
        current_region: torch.Tensor,
        previous_regions: List[torch.Tensor]
    ) -> float:
        """Reward for exploring different regions (avoid repetition)."""
        if not previous_regions:
            return 1.0
        
        overlaps = []
        for prev_region in previous_regions[-3:]:  # Check last 3 regions
            overlap = (current_region * prev_region).sum() / (current_region.sum() + 1e-8)
            overlaps.append(overlap.item())
        
        # Lower overlap = more diversity = higher reward
        avg_overlap = np.mean(overlaps)
        return 1.0 - avg_overlap
    
    def compute_total_reward(
        self,
        **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward from all components.
        
        Returns:
            total_reward: Weighted sum of all components
            reward_breakdown: Individual component values
        """
        reward_breakdown = {}
        total_reward = 0.0
        
        # Diagnosis accuracy
        if 'pred_diagnosis' in kwargs and 'true_diagnosis' in kwargs:
            r = self.compute_diagnosis_reward(
                kwargs['pred_diagnosis'],
                kwargs['true_diagnosis']
            )
            weight = self.components['diagnosis_accuracy']['weight']
            reward_breakdown['diagnosis'] = r
            total_reward += weight * r
        
        # Attention localization
        if 'attention_map' in kwargs and 'saliency_map' in kwargs:
            r = self.compute_attention_localization_reward(
                kwargs['attention_map'],
                kwargs['saliency_map']
            )
            weight = self.components['attention_localization']['weight']
            reward_breakdown['localization'] = r
            total_reward += weight * r
        
        # Reasoning coherence
        if 'reasoning_text' in kwargs:
            r = self.compute_coherence_reward(kwargs['reasoning_text'])
            weight = self.components['reasoning_coherence']['weight']
            reward_breakdown['coherence'] = r
            total_reward += weight * r
        
        # Region relevance
        if 'region_attention' in kwargs:
            r = self.compute_region_relevance_reward(kwargs['region_attention'])
            weight = self.components['region_relevance']['weight']
            reward_breakdown['relevance'] = r
            total_reward += weight * r
        
        # Step diversity
        if 'current_region' in kwargs and 'previous_regions' in kwargs:
            r = self.compute_step_diversity_reward(
                kwargs['current_region'],
                kwargs['previous_regions']
            )
            weight = self.components['step_diversity']['weight']
            reward_breakdown['diversity'] = r
            total_reward += weight * r
        
        # Normalize if specified
        if self.config['reward']['shaping']['normalize']:
            clip_min, clip_max = self.config['reward']['shaping']['clip_range']
            total_reward = np.clip(total_reward, clip_min, clip_max)
        
        return total_reward, reward_breakdown


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for CoT generation.
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        value: ValueNetwork,
        reward_fn: RewardFunction,
        config: dict,
        device: str = 'cuda'
    ):
        self.policy = policy
        self.value = value
        self.reward_fn = reward_fn
        self.config = config['reinforcement_learning']['ppo']
        self.device = device
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config['training']['learning_rate']
        )
        self.value_optimizer = torch.optim.Adam(
            value.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Experience buffer
        self.memory = ExperienceBuffer(capacity=10000)
        
    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        next_values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_values: List of next state value estimates
            dones: List of done flags
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        gamma = self.config.get('gamma', 0.99)
        gae_lambda = self.config.get('gae_lambda', 0.95)
        
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = np.array(advantages)
        returns = np.array(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            batch: Dictionary containing trajectories
            
        Returns:
            metrics: Training metrics
        """
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(batch['log_probs'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch['returns'], dtype=torch.float32, device=self.device)
        
        metrics = {}
        
        # Multiple epochs of updates
        for epoch in range(self.config['n_epochs_per_update']):
            # Forward pass
            action_logits, _ = self.policy(states)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config['clip_epsilon'],
                1.0 + self.config['clip_epsilon']
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_pred = self.value(states).squeeze()
            value_loss = F.mse_loss(value_pred, returns)
            
            # Total loss
            total_loss = (
                policy_loss +
                self.config['value_loss_coef'] * value_loss -
                self.config['entropy_coef'] * entropy
            )
            
            # Update
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config['max_grad_norm']
            )
            torch.nn.utils.clip_grad_norm_(
                self.value.parameters(),
                self.config['max_grad_norm']
            )
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # Metrics
            metrics['policy_loss'] = policy_loss.item()
            metrics['value_loss'] = value_loss.item()
            metrics['entropy'] = entropy.item()
            metrics['total_loss'] = total_loss.item()
        
        return metrics


class ExperienceBuffer:
    """
    Experience replay buffer for RL training.
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Dict):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict:
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = {key: [] for key in self.buffer[0].keys()}
        
        for idx in indices:
            for key, value in self.buffer[idx].items():
                batch[key].append(value)
        
        return batch
    
    def __len__(self):
        return len(self.buffer)

