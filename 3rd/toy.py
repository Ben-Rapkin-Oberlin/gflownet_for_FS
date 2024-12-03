import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleGFlowNet:
    def __init__(self, num_elements=5, target_size=2, hidden_dim=20):
        self.num_elements = num_elements
        self.target_size = target_size
        
        # Simple network architecture
        self.network = nn.Sequential(
            nn.Linear(num_elements, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)

    def get_valid_actions(self, state):
        """Return list of valid actions (elements that can be added)"""
        return [i for i, x in enumerate(state) if x == 0]

    def compute_flow(self, state):
        """Compute flow value for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            return self.network(state_tensor)

    def forward_policy(self, state, possible_actions):
        """Compute action probabilities"""
        flows = []
        for action in possible_actions:
            next_state = state.copy()
            next_state[action] = 1
            flow = self.compute_flow(next_state)
            flows.append(flow)
        
        flows = torch.tensor(flows)
        probs = flows / (flows.sum() + 1e-10)  # Add small constant to avoid division by zero
        return probs

    def sample_subset(self):
        """Sample a subset using the current policy"""
        state = np.zeros(self.num_elements)
        trajectory = []
        
        for _ in range(self.target_size):
            valid_actions = self.get_valid_actions(state)
            if not valid_actions:
                break
                
            # Get probabilities for each action
            probs = self.forward_policy(state, valid_actions)
            
            # Sample action
            action_idx = torch.multinomial(probs, 1).item()
            action = valid_actions[action_idx]
            
            # Update state
            state[action] = 1
            trajectory.append(action)
        
        return trajectory

    def train_step(self, trajectories, rewards):
        """Perform one training step"""
        self.optimizer.zero_grad()
        loss = 0
        
        for trajectory, reward in zip(trajectories, rewards):
            # Convert trajectory to state
            state = np.zeros(self.num_elements)
            for action in trajectory:
                state[action] = 1
            
            # Compute flow for final state
            state_tensor = torch.FloatTensor(state)
            flow = self.network(state_tensor)
            
            # Simple loss: try to make flow match reward
            loss += (flow - reward) ** 2
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Define a simple reward function:
# Prefer selecting elements [1,3] or [2,4]
def toy_reward(subset):
    if sorted(subset) == [1, 3] or sorted(subset) == [2, 4]:
        return torch.tensor(1.0)
    return torch.tensor(0.1)

# Training
model = SimpleGFlowNet(num_elements=5, target_size=2)
num_episodes = 1000
batch_size = 16

print("Starting training...")
for episode in range(num_episodes):
    # Sample batch of trajectories
    trajectories = []
    rewards = []
    
    for _ in range(batch_size):
        traj = model.sample_subset()
        trajectories.append(traj)
        rewards.append(toy_reward(traj))
    
    # Train on batch
    loss = model.train_step(trajectories, rewards)
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Loss: {loss:.4f}")

# Test the trained model
print("\nSampling subsets from trained model:")
for _ in range(5):
    subset = model.sample_subset()
    reward = toy_reward(subset)
    print(f"Sampled subset: {subset}, Reward: {reward.item():.2f}")