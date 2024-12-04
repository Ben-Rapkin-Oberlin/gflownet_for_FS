# utils3.py

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import random
from gflownet3 import ImprovedGFlowNet
from concurrent.futures import ThreadPoolExecutor, as_completed

import cProfile
import pstats

import traceback

# --- Custom time.sleep Wrapper for Debugging ---
# This wrapper will help identify any remaining time.sleep calls.
# Remove or comment out this section once debugging is complete.

# Backup the original time.sleep
original_sleep = time.sleep

def custom_sleep(seconds):
    print(f"time.sleep called with {seconds} seconds")
    traceback.print_stack()
    original_sleep(seconds)

# Override time.sleep with custom_sleep
time.sleep = custom_sleep

# --- End of Custom Wrapper ---

def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_synthetic_dataset(n_samples=1000, n_features=20, n_informative=5, random_state=42):
    """Create a synthetic dataset with informative and noise features."""
    rng = np.random.RandomState(random_state)
    
    informative_features = rng.randn(n_samples, n_informative)
    noise_features = rng.randn(n_samples, n_features - n_informative)
    
    X = np.hstack([informative_features, noise_features])
    y = (informative_features[:, 0] + informative_features[:, 1] > 0).astype(int)
    
    return X, y

def evaluate_single_subset(X_np, y_np, feature_indices):
    """
    Evaluate a single feature subset using Logistic Regression.
    Args:
        X_np: Numpy array of shape (n_samples, n_features)
        y_np: Numpy array of shape (n_samples,)
        feature_indices: List of selected feature indices
    Returns:
        accuracy: Float, accuracy score of the model
    """
    X_selected = X_np[:, feature_indices]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_np, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=100, solver='lbfgs', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def evaluate_feature_subsets_concurrent(X, y, feature_subsets, max_workers=None):
    """
    Evaluate multiple subsets of features in parallel using ThreadPoolExecutor.
    Args:
        X: Torch tensor of shape (n_samples, n_features)
        y: Torch tensor of shape (n_samples,)
        feature_subsets: List of lists, each containing selected feature indices
        max_workers: Maximum number of threads to use (default: number of processors on the machine)
    Returns:
        accuracies: List of accuracy scores for each feature subset
    """
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    
    accuracies = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_subset = {
            executor.submit(evaluate_single_subset, X_np, y_np, subset): subset
            for subset in feature_subsets
        }
        # Collect results as they complete
        for future in as_completed(future_to_subset):
            accuracy = future.result()
            accuracies.append(accuracy)
    return accuracies

def main():
    SEED = 42
    set_all_seeds(SEED)
        
    # Cache device information
    DEVICE_COUNT = torch.cuda.device_count()
    DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE_COUNT > 0 else 'CPU'
    device = torch.device('cuda:0' if DEVICE_COUNT > 0 else 'cpu')
    
    if device.type == 'cuda':
        print(f"Using GPU: {DEVICE_NAME}")
    else:
        print("Using CPU")
        
    # Create synthetic dataset
    n_samples = 1000
    n_features = 20
    n_select = 5
    n_episodes = 21
    batch_size = 64  # Adjust based on your GPU memory
    
    X_np, y_np = create_synthetic_dataset(n_samples=n_samples, n_features=n_features)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
        
    model = ImprovedGFlowNet(
        num_elements=n_features,
        target_size=n_select,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        device=device
    )

    # Initialize scheduler
    dataset_size = batch_size * n_episodes  # Assuming one batch per episode
    model.initialize_scheduler(dataset_size, batch_size, n_episodes)

    # Training loop
    print(f"Training GFlowNet on {device}...")
    
    start_time = time.time()
    best_reward = 0
    
    for episode in range(n_episodes):
        temp = max(1.0 - episode / n_episodes, 0.1)
        
        trajectories = []
        feature_subsets = []
        
        for _ in range(batch_size):
            subset, _ = model.sample_subset(temperature=temp)
            trajectories.append(subset)
            feature_subsets.append(subset)
        
        # Evaluate all subsets in parallel using ThreadPoolExecutor
        accuracies = evaluate_feature_subsets_concurrent(X, y, feature_subsets, max_workers=None)
        accuracies = torch.tensor(accuracies, device=device)
        
        # Update rewards and feature stats
        rewards = []
        for subset, accuracy in zip(trajectories, accuracies):
            rewards.append(accuracy)
            model.update_feature_stats(subset, accuracy.item())
            if accuracy > best_reward:
                best_reward = accuracy.item()
                print(f"New best reward: {best_reward:.4f}")
        
        loss = model.train_step(trajectories, rewards, temp)
        
        if episode % 5 == 0:
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
        subset, _ = model.sample_subset(temperature=0.1)
        accuracy = evaluate_feature_subsets_concurrent(X, y, [subset], max_workers=None)[0]
        final_subsets.append(subset)
        final_accuracies.append(accuracy)
        print(f"Selected features: {subset}, Accuracy: {accuracy:.4f}")
    
    # Print final statistics
    avg_accuracy = np.mean(final_accuracies)
    std_accuracy = np.std(final_accuracies)
    print(f"\nAverage accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    
    # Feature importance analysis
    feature_importance = model.get_feature_importance()
    top_features = np.argsort(feature_importance)[::-1][:n_select]
    print("\nTop features by importance:")
    for idx, feature in enumerate(top_features):
        print(f"Feature {feature}: {feature_importance[feature]:.4f}")

if __name__ == "__main__":
    cProfile.run('main()', 'profile_stats')
    p = pstats.Stats('profile_stats')
    p.sort_stats('tottime').print_stats(10)