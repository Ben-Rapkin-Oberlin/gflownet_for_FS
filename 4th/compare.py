# feature_selection_comparison.py

import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import time
import random
from gflownet3 import ImprovedGFlowNet
from concurrent.futures import ThreadPoolExecutor, as_completed
import cProfile
import traceback

def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_synthetic_dataset(n_samples=1000, n_features=50, n_informative=15, random_state=42):
    """
    Create a synthetic dataset with weaker predictors spread throughout the feature set.
    """
    rng = np.random.RandomState(random_state)
    
    # Generate all features
    X = rng.randn(n_samples, n_features)
    
    # Randomly select informative features
    informative_indices = rng.choice(n_features, size=n_informative, replace=False)
    
    # Assign small coefficients to informative features
    coefficients = np.zeros(n_features)
    coefficients[informative_indices] = rng.uniform(0.1, 0.3, size=n_informative)
    
    # Generate target variable with small contributions from informative features
    y = X @ coefficients + rng.randn(n_samples) * 0.5  # Add some noise
    
    return X, y, informative_indices  # Return informative indices for reference


def evaluate_single_subset(X_np, y_np, feature_indices):
    """
    Evaluate a single feature subset using Linear Regression.
    Args:
        X_np: Numpy array of shape (n_samples, n_features)
        y_np: Numpy array of shape (n_samples,)
        feature_indices: List of selected feature indices
    Returns:
        mse: Float, Mean Squared Error of the model
    """
    X_selected = X_np[:, feature_indices]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_np, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def evaluate_feature_subsets_concurrent(X, y, feature_subsets, max_workers=None):
    """
    Evaluate multiple subsets of features in parallel using ThreadPoolExecutor.
    Args:
        X: Torch tensor of shape (n_samples, n_features)
        y: Torch tensor of shape (n_samples,)
        feature_subsets: List of lists, each containing selected feature indices
        max_workers: Maximum number of threads to use (default: number of processors on the machine)
    Returns:
        mses: List of MSE scores for each feature subset
    """
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    
    mses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_subset = {
            executor.submit(evaluate_single_subset, X_np, y_np, subset): subset
            for subset in feature_subsets
        }
        # Collect results as they complete
        for future in as_completed(future_to_subset):
            mse = future.result()
            mses.append(mse)
    return mses

def sequential_feature_selection(X_np, y_np, m):
    """
    Perform sequential feature selection by removing the least significant variable based on t-values.
    Args:
        X_np: Numpy array of shape (n_samples, n_features)
        y_np: Numpy array of shape (n_samples,)
        m: Number of features to select
    Returns:
        selected_features: List of selected feature indices
        mse: Float, Mean Squared Error of the reduced model
    """
    n_features = X_np.shape[1]
    remaining_features = list(range(n_features))
    X_selected = X_np[:, remaining_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_np, test_size=0.2, random_state=42
    )
    
    while len(remaining_features) > m:
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_train)
        residuals = y_train - y_pred
        X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        params = np.linalg.lstsq(X_design, y_train, rcond=None)[0]
        mse = np.mean(residuals ** 2)
        var_beta = mse * np.linalg.inv(X_design.T @ X_design).diagonal()
        std_errors = np.sqrt(var_beta)
        t_values = params / std_errors
        # Skip the intercept
        t_values = t_values[1:]
        min_t_index = np.argmin(np.abs(t_values))
        # Remove feature with least significant t-value
        del remaining_features[min_t_index]
        X_train = np.delete(X_train, min_t_index, axis=1)
        X_test = np.delete(X_test, min_t_index, axis=1)
    
    # Evaluate the final model
    reg_final = LinearRegression()
    reg_final.fit(X_train, y_train)
    y_pred_test = reg_final.predict(X_test)
    mse_final = mean_squared_error(y_test, y_pred_test)
    
    selected_features = remaining_features
    return selected_features, mse_final

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
    n_features = 100
    n_informative = 0  # Number of informative features
    n_select = 10
    n_episodes = 100
    batch_size = 6  # Adjust based on your GPU memory
        
    X_np, y_np, informative_indices = create_synthetic_dataset(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative
    )
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
        
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
    best_reward = float('inf')
    
    for episode in range(n_episodes):
        temp = max(1.0 - episode / n_episodes, 0.1)
        
        trajectories = []
        feature_subsets = []
        
        for _ in range(batch_size):
            subset, _ = model.sample_subset(temperature=temp)
            trajectories.append(subset)
            feature_subsets.append(subset)
        
        # Evaluate all subsets in parallel using ThreadPoolExecutor
        mses = evaluate_feature_subsets_concurrent(X, y, feature_subsets, max_workers=None)
        mses = torch.tensor(mses, device=device)
        
        # Update rewards and feature stats
        rewards = []
        for subset, mse in zip(trajectories, mses):
            reward = -mse.item()  # Negative MSE as reward
            rewards.append(reward)
            model.update_feature_stats(subset, reward)
            if mse < best_reward:
                best_reward = mse
                print(f"New best reward (lowest MSE): {best_reward:.4f}")
        
        loss = model.train_step(trajectories, rewards, temp)
        
        if episode % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}, Loss: {loss:.4f}, "
                  f"Best Reward (Lowest MSE): {best_reward:.4f}, "
                  f"Temp: {temp:.4f}, Time: {elapsed:.1f}s")
    
    model.load_best_model()
    
    # Final evaluation with GFlowNet-selected features
    print("\nFinal evaluation using best model (GFlowNet-selected features):")
    final_subsets = []
    final_mses = []
    
    for _ in range(10):
        subset, _ = model.sample_subset(temperature=0.1)
        mse = evaluate_feature_subsets_concurrent(X, y, [subset], max_workers=None)[0]
        final_subsets.append(subset)
        final_mses.append(mse)
        print(f"Selected features: {subset}, MSE: {mse:.4f}")
    
    # Print final statistics
    avg_mse = np.mean(final_mses)
    std_mse = np.std(final_mses)
    print(f"\nAverage MSE (GFlowNet): {avg_mse:.4f} ± {std_mse:.4f}")
    
    # Feature importance analysis
    feature_importance = model.get_feature_importance()
    top_features_gflownet = np.argsort(feature_importance)[::-1][:n_select]
    print("\nTop features by importance (GFlowNet):")
    for idx, feature in enumerate(top_features_gflownet):
        print(f"Feature {feature}: {feature_importance[feature]:.4f}")
    
    # Sequential Feature Selection
    print("\nSequential Feature Selection:")
    selected_features_seq, mse_seq = sequential_feature_selection(X_np, y_np, n_select)
    print(f"Selected features: {selected_features_seq}, MSE: {mse_seq:.4f}")
    
    # Comparison
    print("\nComparison of Feature Selection Methods:")
    print(f"GFlowNet Average MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"Sequential Selection MSE: {mse_seq:.4f}")
    
    # Identify overlapping features
    gflownet_features = set(top_features_gflownet)
    sequential_features = set(selected_features_seq)
    common_features = gflownet_features & sequential_features
    print(f"\nCommon features between GFlowNet and Sequential Selection: {common_features}")
    print(f"Unique to GFlowNet: {gflownet_features - sequential_features}")
    print(f"Unique to Sequential Selection: {sequential_features - gflownet_features}")

if __name__ == "__main__":
    # Run the main function under cProfile for profiling
    main()
