import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads,
            dropout=0.1,  # Add dropout to attention weights
            batch_first=True  # Optimize memory layout
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        attn_out, weights = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        return self.norm(x + attn_out), weights

class ImprovedGFlowNet(nn.Module):
    def __init__(self, num_elements, target_size, hidden_dim=128, num_heads=4, 
                 num_layers=3, dropout_rate=0.1, device='cuda'):
        super().__init__()
        self.num_elements = num_elements
        self.target_size = target_size
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        
        # Use parameter groups for different learning rates
        self.feature_params = nn.ParameterList()
        self.attention_params = nn.ParameterList()
        self.output_params = nn.ParameterList()

        # Add running statistics for reward normalization
        self.register_buffer('reward_mean', torch.zeros(1))
        self.register_buffer('reward_std', torch.ones(1))
        self.reward_momentum = 0.99
        
        # Efficient embedding with weight sharing
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.feature_params.extend(self.feature_embedding.parameters())
        
        # Improved positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(num_elements, hidden_dim))
        
        # Memory-efficient attention layers with gradient checkpointing
        self.attention_layers = nn.ModuleList([
            FeatureAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        for layer in self.attention_layers:
            self.attention_params.extend(layer.parameters())
        
        # Efficient final layers with bottleneck
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * num_elements, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        self.output_params.extend(self.final_layers.parameters())
        
        # Initialize optimizers with different learning rates
        self.optimizer = self._create_optimizer()
        self.scaler = GradScaler()
        
        # Feature importance tracking with memory-efficient storage
        self.register_buffer('feature_counts', torch.zeros(num_elements))
        self.register_buffer('feature_rewards', torch.zeros(num_elements))
        self.selection_history = []
        
        self.to(self.device)
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create more efficient sinusoidal positional encodings"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
    
    def _create_optimizer(self):
        """Create optimizer with different parameter groups"""
        return optim.AdamW([
            {'params': self.feature_params, 'lr': 0.003},
            {'params': self.attention_params, 'lr': 0.002},
            {'params': self.output_params, 'lr': 0.003}
        ], weight_decay=0.01)

    @torch.amp.autocast('cuda')  # Updated decorator
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        
        # Efficient embedding
        x = self.feature_embedding(x)
        x = x + self.pos_encoding.unsqueeze(0)
        
        # Process through attention layers with gradient checkpointing
        attention_weights = []
        for attention in self.attention_layers:
            # Add use_reentrant=False parameter
            x, weights = torch.utils.checkpoint.checkpoint(attention, x, use_reentrant=False)
            attention_weights.append(weights)
        
        # Memory-efficient reshape and final layer processing
        x = x.reshape(batch_size, -1)
        flows = self.final_layers(x)
        
        return flows, attention_weights


    def forward_policy(self, state, possible_actions, temperature=1.0):
        """Memory-efficient policy computation"""
        if len(possible_actions) == 0:
            return torch.tensor([]).to(self.device)
        
        # Ensure all tensors are on the correct device
        indices = torch.stack([
            torch.arange(len(possible_actions), device=self.device).repeat_interleave(self.num_elements),
            torch.arange(self.num_elements, device=self.device).repeat(len(possible_actions))
        ])
        values = state.repeat(len(possible_actions))
        
        # Use sparse operations for state manipulation
        next_states = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(len(possible_actions), self.num_elements),
            device=self.device
        ).to_dense()
        
        next_states[torch.arange(len(possible_actions), device=self.device), possible_actions] = 1
        
        with torch.no_grad(), autocast():
            flows, _ = self(next_states)
            logits = torch.log(flows.squeeze() + 1e-8) / temperature
            probs = F.softmax(logits, dim=0)
        
        return probs

    def train_step(self, trajectories, rewards, temperature):
        """Improved training step with gradient accumulation"""
        self.optimizer.zero_grad()
        
        # Process in smaller batches to save memory
        batch_size = 36
        total_loss = 0
        num_batches = (len(trajectories) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(trajectories))
            
            batch_trajectories = trajectories[start_idx:end_idx]
            batch_rewards = rewards[start_idx:end_idx]
            
            states = torch.zeros(len(batch_trajectories), self.num_elements, device=self.device)
            for idx, trajectory in enumerate(batch_trajectories):
                states[idx, trajectory] = 1
            
            batch_rewards = torch.tensor(batch_rewards, device=self.device).unsqueeze(1)
            # Update running statistics and normalize rewards
            with torch.no_grad():
                batch_mean = batch_rewards.mean()
                batch_std = batch_rewards.std() + 1e-8
                
                # Update running statistics
                self.reward_mean = self.reward_momentum * self.reward_mean + (1 - self.reward_momentum) * batch_mean
                self.reward_std = self.reward_momentum * self.reward_std + (1 - self.reward_momentum) * batch_std
                
            # Normalize using running statistics instead of batch statistics
            batch_rewards = (batch_rewards -self.reward_mean) / (self.reward_std + 1e-8)

            with autocast():
                flows, _ = self(states)
                loss = F.mse_loss(flows, batch_rewards)
                # Add L2 regularization selectively
                l2_reg = sum(p.pow(2.0).sum() for p in self.attention_params)
                loss = loss + 1e-5 * l2_reg
                loss = loss / num_batches  # Scale loss for gradient accumulation
            
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
        
        # Clip gradients and optimize
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_loss

    def update_feature_stats(self, subset, reward):
        """Memory-efficient feature statistics update"""
        subset_tensor = torch.tensor(subset, device=self.device)
        self.feature_counts.index_add_(0, subset_tensor, torch.ones_like(subset_tensor, dtype=torch.float))
        self.feature_rewards.index_add_(0, subset_tensor, torch.full_like(subset_tensor, reward, dtype=torch.float))
        
        # Keep only recent history to save memory
        MAX_HISTORY = 1000
        self.selection_history.append((subset, reward))
        if len(self.selection_history) > MAX_HISTORY:
            self.selection_history = self.selection_history[-MAX_HISTORY:]

    def get_feature_importance(self):
        """Compute feature importance scores efficiently"""
        with torch.no_grad():
            counts = self.feature_counts.clone()
            counts[counts == 0] = 1  # Avoid division by zero
            importance_scores = (self.feature_rewards / counts).cpu().numpy()
        return importance_scores
        
    def get_valid_actions(self, state):
        """Get valid actions efficiently using masked operations"""
        with torch.no_grad():
            # Use mask operations instead of nonzero for efficiency
            mask = (state == 0)
            if state.sum() >= self.target_size:
                return torch.tensor([], dtype=torch.long, device=self.device)
            return mask.nonzero(as_tuple=True)[0]

    @torch.no_grad()  # Memory optimization
    def sample_subset(self, temperature=1.0):
        """
        Sample a subset of features efficiently using the GFlowNet policy.
        Args:
            temperature: Float, controls exploration (higher = more exploration)
        Returns:
            trajectory: List of selected feature indices
            attention_weights: List of attention weights
        """
        # Initialize state tensor efficiently
        state = torch.zeros(self.num_elements, device=self.device)
        trajectory = []
        attention_weights = []
        
        # Clamp temperature for stability
        temperature = max(min(temperature, 5.0), 0.1)
        
        for _ in range(self.target_size):
            valid_actions = self.get_valid_actions(state)
            if len(valid_actions) == 0:
                break
            
            # Compute policy probabilities
            probs = self.forward_policy(state, valid_actions, temperature)
            if len(probs) == 0:
                break
            
            # Sample action efficiently
            action_idx = torch.multinomial(probs, 1).item()
            action = valid_actions[action_idx].item()
            
            # Update state using in-place operation
            state[action] = 1
            trajectory.append(action)
            
            # Get attention weights if needed (memory efficient)
            if len(attention_weights) < len(self.attention_layers):
                with autocast():
                    _, weights = self(state.unsqueeze(0))
                    attention_weights.append([w[-1:] for w in weights])  # Keep only last step
        
        return trajectory, attention_weights