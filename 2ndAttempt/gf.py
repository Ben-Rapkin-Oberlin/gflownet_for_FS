from feature_selection_gfn import (
    FeatureSelectionEnv,
    create_feature_selection_gfn,
    train_feature_selection_gfn
)
import torch

# Example reward function - replace with your own
def reward_fn(selected_features):
    target_features = set([0, 2, 4])
    selected_features = set(selected_features)
    overlap = len(target_features.intersection(selected_features))
    return float(overlap) / len(target_features)

# Create environment
env = FeatureSelectionEnv(
    n_features=10,  # Total number of features
    target_size=3,  # Number of features to select
    reward_fn=reward_fn,
)

# Create and train GFlowNet
device = "cuda" if torch.cuda.is_available() else "cpu"
gflownet = create_feature_selection_gfn(env, device)
gflownet = train_feature_selection_gfn(
    gflownet, 
    env, 
    n_iterations=1000,
    batch_size=32,
    device=device
)

# Generate some feature selections
with torch.no_grad():
    trajectories = gflownet.sample_trajectories(env, n=5)
    for traj in trajectories:
        selected = traj.last_states.selected_features
        reward = torch.exp(env.log_reward(traj.last_states)).item()
        print(f"Selected features: {selected}, Reward: {reward:.4f}")