# improved_gflownet.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FeatureAttention(nn.Module):
    """
    An attention layer that processes per-feature embeddings.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (seq_length, batch_size, embed_dim)
        attn_out, weights = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        return self.norm(x + attn_out), weights  # Residual connection

class ImprovedGFlowNet(nn.Module):
    """
    An improved GFlowNet architecture for feature selection with attention mechanisms.
    """
    def __init__(self, num_elements, target_size, hidden_dim=128, num_heads=4, 
                 num_layers=3, dropout_rate=0.1, device='cuda'):
        super().__init__()
        self.num_elements = num_elements
        self.target_size = target_size
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        
        # Per-feature embedding
        self.feature_embedding = nn.Linear(1, hidden_dim)
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(num_elements, hidden_dim))
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            FeatureAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * num_elements, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive outputs
        )
        
        # Move model to specified device
        self.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Scheduler (will initialize later with correct steps_per_epoch)
        self.scheduler = None
        
        self.best_loss = float('inf')
        self.best_state_dict = None
        
        # Initialize tracking metrics
        self.feature_counts = torch.zeros(num_elements, device=self.device)
        self.feature_rewards = torch.zeros(num_elements, device=self.device)
        self.selection_history = []
        
        # For mixed precision training
        self.scaler = torch.amp.GradScaler()
        
    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: Tensor of shape (batch_size, num_elements), where each element is 0 or 1 indicating feature selection status.
        Returns:
            flows: Tensor of shape (batch_size, 1), representing the flow values.
            attention_weights: List of attention weight tensors from each layer.
        """
        # x is expected to be already on the correct device
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        # Reshape x to (batch_size, num_elements, 1)
        x = x.unsqueeze(-1)
        
        # Apply per-feature embedding
        x = self.feature_embedding(x)  # (batch_size, num_elements, hidden_dim)
        
        # Add positional encoding
        pos_encoding = self.pos_encoding.unsqueeze(0)  # (1, num_elements, hidden_dim)
        x = x + pos_encoding  # (batch_size, num_elements, hidden_dim)
        
        # Transpose x for MultiheadAttention: (seq_length, batch_size, embed_dim)
        x = x.transpose(0, 1)  # (num_elements, batch_size, hidden_dim)
        
        # Process through attention layers
        attention_weights = []
        for attention in self.attention_layers:
            x, weights = attention(x)
            attention_weights.append(weights)
        
        # Transpose back to (batch_size, num_elements, hidden_dim)
        x = x.transpose(0, 1)
        
        # Flatten and pass through final layers
        x = x.reshape(batch_size, -1)
        flows = self.final_layers(x)  # (batch_size, 1)
        
        return flows, attention_weights

    def forward_policy(self, state, possible_actions, temperature=1.0):
        """
        Compute action probabilities for the given state and possible actions.
        Args:
            state: Tensor of shape (num_elements,), current state of feature selection.
            possible_actions: Tensor of indices of possible actions (features to select next).
            temperature: Float, controls exploration (higher temperature -> more exploration).
        Returns:
            probs: Tensor of shape (len(possible_actions),), action probabilities.
        """
        if len(possible_actions) == 0:
            return torch.tensor([]).to(self.device)
        
        # Create tensor of next states
        next_states = state.unsqueeze(0).repeat(len(possible_actions), 1)
        next_states[range(len(possible_actions)), possible_actions] = 1
        
        with torch.no_grad():
            flows, _ = self(next_states)
            # Compute logits directly
            logits = torch.log(flows.squeeze() + 1e-8) / temperature
            probs = torch.softmax(logits, dim=0)
        
        return probs

    def get_valid_actions(self, state):
        """
        Get list of valid actions given the current state.
        Args:
            state: Tensor of shape (num_elements,), current state.
        Returns:
            Tensor of indices of valid actions.
        """
        selected_count = state.sum().item()
        if selected_count >= self.target_size:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return (state == 0).nonzero(as_tuple=True)[0]

    def sample_subset(self, temperature=1.0):
        """
        Sample a subset of features using the GFlowNet policy.
        Args:
            temperature: Float, controls exploration.
        Returns:
            trajectory: List of selected feature indices.
            attention_weights: List of attention weights from each selection step.
        """
        state = torch.zeros(self.num_elements, device=self.device)
        trajectory = []
        attention_weights = []
        
        temperature = max(temperature, 0.1)
        
        for _ in range(self.target_size):
            valid_actions = self.get_valid_actions(state)
            if len(valid_actions) == 0:
                break
            
            probs = self.forward_policy(state, valid_actions, temperature)
            if len(probs) == 0:
                break
            
            action_idx = torch.multinomial(probs, 1).item()
            action = valid_actions[action_idx].item()
            
            state[action] = 1
            trajectory.append(action)
        
        return trajectory, attention_weights

    def train_step(self, trajectories, rewards, temperature):
        """
        Perform a training step using the given trajectories and rewards with mixed precision.
        Args:
            trajectories: List of trajectories (each trajectory is a list of feature indices).
            rewards: List of reward values corresponding to each trajectory.
            temperature: Float, controls exploration.
        Returns:
            loss_value: Float, the computed loss value.
        """
        self.optimizer.zero_grad()
        
        # Prepare states tensor directly on device
        batch_size = len(trajectories)
        states = torch.zeros(batch_size, self.num_elements, device=self.device)
        for idx, trajectory in enumerate(trajectories):
            states[idx, trajectory] = 1
        
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1)
        
        with torch.amp.autocast(device_type='cuda'):
            flows, _ = self(states)
            main_loss = ((flows - rewards) ** 2).mean()
            l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
            loss = main_loss + 1e-5 * l2_reg
        
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Save best model
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in self.state_dict().items()
            }
        
        return loss.item()

    def update_feature_stats(self, subset, reward):
        """
        Update feature selection statistics based on the subset and reward.
        Args:
            subset: List of selected feature indices.
            reward: Float, the reward obtained for the subset.
        """
        for feature in subset:
            self.feature_counts[feature] += 1
            self.feature_rewards[feature] += reward
        self.selection_history.append((subset, reward))

    def get_feature_importance(self):
        """
        Calculate feature importance scores based on selection frequencies and rewards.
        Returns:
            importance_scores: Numpy array of feature importance scores.
        """
        # Calculate average rewards per feature
        total_counts = self.feature_counts.cpu().numpy()
        total_rewards = self.feature_rewards.cpu().numpy()
        
        # Avoid division by zero
        total_counts[total_counts == 0] = 1
        importance_scores = total_rewards / total_counts
        
        return importance_scores

    def load_best_model(self):
        """
        Load the best saved model state.
        """
        if self.best_state_dict is not None:
            self.load_state_dict(self.best_state_dict)

    def initialize_scheduler(self, dataset_size, batch_size, epochs):
        """
        Initialize the learning rate scheduler with the correct steps_per_epoch.
        Args:
            dataset_size: Integer, total number of samples in the dataset.
            batch_size: Integer, batch size used during training.
            epochs: Integer, total number of epochs for training.
        """
        steps_per_epoch = max(dataset_size // batch_size, 1)
        total_steps = steps_per_epoch * epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,
            total_steps=total_steps,
            pct_start=0.1
        )
