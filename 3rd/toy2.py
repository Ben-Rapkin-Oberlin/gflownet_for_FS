import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def create_synthetic_dataset(n_samples=1000, n_features=20, n_informative=5, random_state=42):

    rng = np.random.RandomState(random_state)

    informative_features = rng.randn(n_samples, n_informative)

    noise_features = rng.randn(n_samples, n_features - n_informative)

    X = np.hstack([informative_features, noise_features])

    y = (informative_features[:, 0] + informative_features[:, 1] > 0).astype(int)

    return X, y



def evaluate_feature_subset(X, y, feature_indices, scaler=None, clf=None):
    """Fast evaluation using pre-initialized objects"""
    X_selected = X[:, feature_indices]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    if scaler is None:
        scaler = StandardScaler()
        
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if clf is None:
        clf = LogisticRegression(max_iter=100, solver='lbfgs')
        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def set_all_seeds(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

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

def main():
    SEED = 42
    set_all_seeds(SEED)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic dataset and model
    n_samples = 1000
    n_features = 20
    n_select = 5
    n_episodes = 500
    batch_size = 32  # Increased batch size for GPU
    
    X, y = create_synthetic_dataset(n_samples=n_samples, n_features=n_features)
    
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=100, solver='lbfgs', random_state=SEED)
    
    model = ImprovedGFlowNet(
        num_elements=n_features,
        target_size=n_select,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        device=device
    )
    
    # Training loop with improvements
    print(f"Training GFlowNet on {device}...")
    
    start_time = time.time()
    best_reward = 0
    
    for episode in range(n_episodes):
        temp = max(1.0 - episode/n_episodes, 0.1)
        
        trajectories = []
        rewards = []
        
        # Use beam search occasionally
        use_beam_search = episode % 10 == 0
        
        for _ in range(batch_size):
            subset = model.sample_subset(
                temperature=temp,
                use_beam_search=use_beam_search
            )
            accuracy = evaluate_feature_subset(X, y, subset, scaler, clf)
            trajectories.append(subset)
            rewards.append(torch.tensor(accuracy, device=device))
            model.update_feature_stats(subset, accuracy)
            
            if accuracy > best_reward:
                best_reward = accuracy
                print(f"New best reward: {best_reward:.4f}")
        
        loss = model.train_step(trajectories, rewards, temp)
        
        if episode % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}, Loss: {loss:.4f}, "
                  f"Best Reward: {best_reward:.4f}, "
                  f"Temp: {temp:.4f}, Time: {elapsed:.1f}s")
    
    model.load_best_model()
    
    # Final evaluation
    print("\nFinal evaluation using best model:")
    final_subsets = []
    final_accuracies = []
    
    for _ in range(10):
        subset = model.sample_subset(temperature=0.1, use_beam_search=True)
        accuracy = evaluate_feature_subset(X, y, subset, scaler, clf)
        final_subsets.append(subset)
        final_accuracies.append(accuracy)
        print(f"Selected features: {subset}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()