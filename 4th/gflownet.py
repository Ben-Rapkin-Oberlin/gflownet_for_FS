import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)  # Added dropout for regularization
        
    def forward(self, x):
        attn_out, weights = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        return self.norm(x + attn_out), weights



class ImprovedGFlowNet(nn.Module):
    def __init__(self, num_elements, target_size, hidden_dim=128, num_heads=4, 
                     num_layers=3, dropout_rate=0.1, device='cuda'):
            super().__init__()
            self.num_elements = num_elements
            self.target_size = target_size
            self.device = device
            self.hidden_dim = hidden_dim
            
            # Feature embedding - process all features at once
            self.feature_embedding = nn.Sequential(
                nn.Linear(num_elements, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            
            # Position encoding
            self.pos_encoding = nn.Parameter(torch.randn(1, num_elements, hidden_dim))
            
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
                nn.Sigmoid()
            )
            
            # Move model to specified device
            self.to(device)
            
            # Optimizer
            self.optimizer = optim.AdamW(
                self.parameters(),
                lr=0.001,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
            
            # Scheduler
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=0.001,
                epochs=50,
                steps_per_epoch=100,
                pct_start=0.1
            )
            
            self.best_loss = float('inf')
            self.best_state_dict = None
            
            # Initialize tracking metrics
            self.feature_counts = torch.zeros(num_elements, device=device)
            self.feature_rewards = torch.zeros(num_elements, device=device)
            self.selection_history = []
            
    def forward(self, x):
        # Move input to device and ensure correct shape
        x = x.to(self.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
    
        batch_size = x.size(0)
    
        # Feature embedding
        x = self.feature_embedding(x)  # (batch_size, hidden_dim)
        x = x.unsqueeze(1).expand(-1, self.num_elements, -1)  # (batch_size, num_elements, hidden_dim)
    
        # Add positional encoding
        pos_encoding = self.pos_encoding.expand(batch_size, -1, -1)
        x = x + pos_encoding
    
        # Process through attention layers
        attention_weights = []
        for attention in self.attention_layers:
            x, weights = attention(x)
            attention_weights.append(weights)
    
        # Final processing
        x = x.reshape(batch_size, -1)
        return self.final_layers(x), attention_weights

    def forward_policy(self, state, possible_actions, temperature=1.0):
        """Compute action probabilities with improved numerical stability"""
        if not possible_actions:
            return torch.tensor([]).to(self.device)
    
        # Create tensor of next states more efficiently
        next_states = []
        for action in possible_actions:
            next_state = state.copy()
            next_state[action] = 1
            next_states.append(next_state)
            
        # Convert to tensor with proper dimensions
        next_states = np.array(next_states)
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        # Ensure proper shape (batch_size, num_features)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(0)
    
        with torch.no_grad():
            flows, _ = self(next_states)
            
            # Improved numerical stability
            log_flows = torch.log(flows + 1e-10)
            log_probs = log_flows / temperature
            probs = torch.softmax(log_probs, dim=0)
            
            # Ensure we return a 1D tensor
            if len(probs.shape) > 1:
                probs = probs.squeeze()
        
        return probs

    def get_valid_actions(self, state):
        """Get list of valid actions given current state"""
        selected_count = np.sum(state)
        if selected_count >= self.target_size:
            return []
        return [i for i in range(self.num_elements) if state[i] == 0]

    def sample_subset(self, temperature=1.0, use_beam_search=False, beam_width=3):
        """Sample a subset using either standard sampling or beam search"""
        if use_beam_search:
            return self._beam_search_sample(beam_width)
        
        state = np.zeros(self.num_elements)
        trajectory = []
        attention_weights = []
        
        temperature = max(temperature, 0.1)
        
        for _ in range(self.target_size):
            valid_actions = self.get_valid_actions(state)
            if not valid_actions:
                break
                
            probs = self.forward_policy(state, valid_actions, temperature)
            
            try:
                if random.random() < 0.1:  # Exploration
                    action_idx = random.randrange(len(valid_actions))
                else:
                    action_idx = torch.multinomial(probs, 1).item()
                action = valid_actions[action_idx]
            except RuntimeError:
                action_idx = np.random.randint(len(valid_actions))
                action = valid_actions[action_idx]
            
            state[action] = 1
            trajectory.append(action)
            
            # Store attention weights
            with torch.no_grad():
                _, weights = self(torch.FloatTensor(state).to(self.device))
                attention_weights.append(weights)
        
        return trajectory, attention_weights

    def _beam_search_sample(self, beam_width):
        """Implement beam search for more stable sampling"""
        beams = [(np.zeros(self.num_elements), [], 1.0)]
        attention_weights_history = []
        
        for _ in range(self.target_size):
            candidates = []
            for state, trajectory, score in beams:
                valid_actions = self.get_valid_actions(state)
                if not valid_actions:
                    continue
                    
                probs = self.forward_policy(state, valid_actions, temperature=0.1)
                
                if len(probs.shape) == 0:  # Handle scalar output
                    probs = probs.unsqueeze(0)
                
                for action_idx, prob in enumerate(probs):
                    if action_idx >= len(valid_actions):
                        continue
                    action = valid_actions[action_idx]
                    new_state = state.copy()
                    new_state[action] = 1
                    new_score = score * prob.item()
                    candidates.append((
                        new_state,
                        trajectory + [action],
                        new_score
                    ))
            
            if not candidates:
                break
                
            # Select top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]
            
            # Store attention weights for best beam
            with torch.no_grad():
                _, weights = self(torch.FloatTensor(beams[0][0]).to(self.device))
                attention_weights_history.append(weights)
        
        # Return the best trajectory and attention weights
        return beams[0][1], attention_weights_history

    def train_step(self, trajectories, rewards, temperature):
        """Enhanced training step with various improvements"""
        self.optimizer.zero_grad()
        
        # Move data to device
        states = torch.FloatTensor([
            [1 if i in trajectory else 0 for i in range(self.num_elements)]
            for trajectory in trajectories
        ]).to(self.device)
        
        rewards = torch.stack(rewards).to(self.device)
        
        # Forward pass
        flows, attention_weights = self(states)
        
        # Multiple loss components
        main_loss = ((flows - rewards) ** 2).mean()
        
        # L2 regularization
        l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
        
        # Attention diversity loss
        attention_loss = 0
        for weights in attention_weights:
            attention_loss += -torch.mean(
                torch.sum(weights * torch.log(weights + 1e-10), dim=-1)
            )
        
        # Combined loss
        loss = main_loss + 1e-5 * l2_reg + 1e-3 * attention_loss
        
        # Gradient clipping and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Save best model
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in self.state_dict().items()
            }
        
        return loss.item()

    def update_feature_stats(self, subset, reward):
        """Update feature selection statistics"""
        for feature in subset:
            self.feature_counts[feature] += 1
            self.feature_rewards[feature] += reward
        self.selection_history.append((subset, reward))

    def get_feature_importance(self):
        """Calculate feature importance scores with attention weights"""
        avg_rewards = np.zeros(self.num_elements)
        total_counts = np.zeros(self.num_elements)
        
        with torch.no_grad():
            for subset, reward in self.selection_history[-100:]:  # Use recent history
                state = torch.zeros(self.num_elements, device=self.device)
                state[subset] = 1
                _, attention_weights = self(state.unsqueeze(0))
                
                # Calculate average attention per feature
                avg_attention = torch.mean(
                    torch.stack([w.mean(0).squeeze() for w in attention_weights]), 
                    dim=0
                ).cpu().numpy()
                
                # Update rewards for selected features only
                for idx, feature in enumerate(subset):
                    avg_rewards[feature] += reward
                    total_counts[feature] += 1
        
        # Avoid division by zero
        total_counts[total_counts == 0] = 1
        return avg_rewards / total_counts


    def load_best_model(self):
        """Load the best model state dict"""
        if self.best_state_dict is not None:
            self.load_state_dict(self.best_state_dict)