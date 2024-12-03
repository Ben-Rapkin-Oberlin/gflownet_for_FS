import torch

# Example reward function for subset selection
def reward_function(subset):
    """
    Example reward function that prefers selecting certain elements.
    You would replace this with your specific objective.
    """
    # Example: prefer selecting elements with indices 2, 4, 7
    preferred = torch.tensor([2, 4, 7])
    reward = sum(1 for i in preferred if subset[i] == 1)
    return torch.tensor(reward, dtype=torch.float32)

# Initialize GFlowNet for selecting 3 elements from a set of 10
model = SetSelectionGFlowNet(num_elements=10, target_size=3)

# Training loop
num_episodes = 40000  # As used in paper
batch_size = 32

for episode in range(num_episodes):
    # Sample batch of trajectories
    trajectories = []
    rewards = []
    
    for _ in range(batch_size):
        initial_state = torch.zeros(10)
        trajectory = model.sample_trajectory(initial_state, max_steps=3)
        final_subset = torch.zeros(10)
        for _, action in trajectory:
            final_subset[action] = 1
        
        trajectories.append(trajectory)
        rewards.append(reward_function(final_subset))
    
    # Train on batch
    loss = model.train_step(trajectories, rewards)
    
    if episode % 1000 == 0:
        print(f"Episode {episode}, Loss: {loss:.4f}")

# Sample subsets using trained model
sampled_subsets = model.sample_subset(reward_function, num_samples=5)
for subset in sampled_subsets:
    print("Sampled subset:", torch.nonzero(subset).squeeze().tolist())