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
        
        # Improved feature embedding with position encoding
        self.embedding = nn.Sequential(
            nn.Linear(num_elements, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, num_elements, hidden_dim))
        
        # Multiple attention layers with residual connections
        self.attention_layers = nn.ModuleList([
            FeatureAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Improved final layers with skip connections
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Doubled input size for skip connection
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Move model to specified device
        self.to(device)
        
        # Improved optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=0.001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.best_loss = float('inf')
        self.best_state_dict = None
        
        # Enhanced tracking metrics
        self.feature_counts = np.zeros(num_elements)
        self.feature_rewards = np.zeros(num_elements)
        self.selection_history = []
        self.attention_weights_history = []

    def forward(self, x):
        # Move input to device
        x = x.to(self.device)
        
        # Initial embedding with position encoding
        x = self.embedding(x)
        x = x.unsqueeze(0) + self.pos_encoding
        
        # Store attention weights for analysis
        attention_weights = []
        
        # Process through attention layers with residual connections
        initial_x = x
        for attention in self.attention_layers:
            x, weights = attention(x)
            attention_weights.append(weights)
            
        x = x.squeeze(0)
        
        # Skip connection from initial embedding
        x_combined = torch.cat([x, initial_x.squeeze(0)], dim=-1)
        
        return self.final_layers(x_combined), attention_weights

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
            
        # Convert to numpy array first, then to tensor
        next_states = np.array(next_states)
        next_states = torch.FloatTensor(next_states).to(self.device)
    
        with torch.no_grad():
            flows, _ = self(next_states)
            
            # Improved numerical stability
            log_flows = torch.log(flows + 1e-10)
            log_probs = log_flows / temperature
            probs = torch.softmax(log_probs, dim=0)
        
        return probs.squeeze()

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
        
        for _ in range(self.target_size):
            candidates = []
            for state, trajectory, score in beams:
                valid_actions = self.get_valid_actions(state)
                if not valid_actions:
                    continue
                    
                probs = self.forward_policy(state, valid_actions, temperature=0.1)
                
                for action_idx, prob in enumerate(probs):
                    action = valid_actions[action_idx]
                    new_state = state.copy()
                    new_state[action] = 1
                    new_score = score * prob.item()
                    candidates.append((
                        new_state,
                        trajectory + [action],
                        new_score
                    ))
            
            # Select top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]
        
        # Return the best trajectory
        return beams[0][1]

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
        with torch.no_grad():
            for subset, reward in self.selection_history[-100:]:  # Use recent history
                state = torch.zeros(self.num_elements).to(self.device)
                state[subset] = 1
                _, attention_weights = self(state.unsqueeze(0))
                
                # Aggregate attention weights
                avg_attention = torch.mean(
                    torch.cat([w.mean(0) for w in attention_weights]), dim=0
                ).cpu().numpy()
                
                avg_rewards[subset] += reward * avg_attention
        
        return avg_rewards / max(1, len(self.selection_history[-100:]))

    def load_best_model(self):
        """Load the best model state dict"""
        if self.best_state_dict is not None:
            self.load_state_dict(self.best_state_dict)