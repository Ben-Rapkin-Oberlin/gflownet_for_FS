# feature_selection_comparison.py

import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import time
import random
import matplotlib.pyplot as plt
from gflownet3 import ImprovedGFlowNet
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import cProfile
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_synthetic_dataset(
    n_samples=1000,
    n_features=50,
    n_informative=2,
    n_multicollinear=3,
    correlation_strength=0.7,  # Correlation strength of informative features
    noise_level=0.1,          # Amount of noise in the target variable
    multicollinear_noise=0.1, # Noise in multicollinear relationships
    random_state=42
):
    """
    Create a synthetic dataset with controlled feature relationships.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Total number of features
    n_informative : int
        Number of features that directly influence the target
    n_multicollinear : int
        Number of features that are correlated with each informative feature
    correlation_strength : float
        How strongly the informative features correlate with target (0 to 1)
    noise_level : float
        Standard deviation of noise added to target
    multicollinear_noise : float
        Noise added to multicollinear relationships
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        The feature matrix
    y : ndarray of shape (n_samples,)
        The target values
    feature_types : dict
        Dictionary containing indices of different feature types
    """
    rng = np.random.RandomState(random_state)
    
    # Initialize feature matrix with random noise
    X = rng.randn(n_samples, n_features)
    
    # Create informative features
    informative_indices = list(range(n_informative))
    informative_features = rng.randn(n_samples, n_informative)
    
    # Create target variable with controlled correlation
    y = np.zeros(n_samples)
    for i in range(n_informative):
        y += correlation_strength * informative_features[:, i]
    
    # Add noise to target
    y += rng.normal(0, noise_level, n_samples)
    
    # Create multicollinear features
    multicollinear_indices = []
    current_idx = n_informative
    
    for inf_idx in range(n_informative):
        related_indices = list(range(current_idx, current_idx + n_multicollinear))
        multicollinear_indices.extend(related_indices)
        
        # Create features correlated with each informative feature
        for j in range(n_multicollinear):
            correlation = rng.uniform(0.7, 0.9)  # Random high correlation
            noise = rng.normal(0, multicollinear_noise, n_samples)
            X[:, related_indices[j]] = (correlation * informative_features[:, inf_idx] + 
                                      np.sqrt(1 - correlation**2) * noise)
        
        current_idx += n_multicollinear
    
    # Place informative features in the matrix
    X[:, informative_indices] = informative_features
    
    # Track feature types
    feature_types = {
        'informative': informative_indices,
        'multicollinear': multicollinear_indices,
        'noise': list(set(range(n_features)) - 
                     set(informative_indices) - 
                     set(multicollinear_indices))
    }
    
    # Standardize all features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    
    return X, y, feature_types


def evaluate_single_subset(X_np, y_np, feature_indices):
    """
    Evaluate a single feature subset using Linear Regression.
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
    """
    # Move tensors to CPU and convert to numpy
    if torch.is_tensor(X):
        X_np = X.cpu().numpy()
    else:
        X_np = X
        
    if torch.is_tensor(y):
        y_np = y.cpu().numpy()
    else:
        y_np = y
    
    mses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_subset = {
            executor.submit(evaluate_single_subset, X_np, y_np, subset): subset
            for subset in feature_subsets
        }
        for future in as_completed(future_to_subset):
            try:
                mse = future.result()
                mses.append(mse)
            except Exception as e:
                print(f"Error evaluating subset {future_to_subset[future]}: {e}")
    return mses

def sequential_feature_selection(X_np, y_np, m):
    """
    Perform sequential feature selection.
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
        t_values = t_values[1:]  # Skip the intercept
        min_t_index = np.argmin(np.abs(t_values))
        del remaining_features[min_t_index]
        X_train = np.delete(X_train, min_t_index, axis=1)
        X_test = np.delete(X_test, min_t_index, axis=1)
    
    reg_final = LinearRegression()
    reg_final.fit(X_train, y_train)
    y_pred_test = reg_final.predict(X_test)
    mse_final = mean_squared_error(y_test, y_pred_test)
    
    return remaining_features, mse_final


def analyze_feature_selection(selected_features, feature_types):
    """Analyze which types of features were selected."""
    n_informative = len(set(selected_features) & set(feature_types['informative']))
    n_multicollinear = len(set(selected_features) & set(feature_types['multicollinear']))
    n_noise = len(set(selected_features) & set(feature_types['noise']))
    
    return {
        'n_informative': n_informative,
        'n_multicollinear': n_multicollinear,
        'n_noise': n_noise
    }

def ensure_cpu(val):
    """Ensure the value is on CPU and converted from tensor if necessary."""
    if torch.is_tensor(val):
        return val.cpu().item()
    return val

def visualize_results(results, feature_types):
    """Create visualization of results."""
    # Get method names and their metrics
    methods = list(results.keys())
    mses = [ensure_cpu(data['mse']) for data in results.values()]
    analyses = [analyze_feature_selection(data['features'], feature_types) for data in results.values()]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[1, 1.2])

    # MSE Plot (top)
    bars = ax1.bar(methods, mses, color='skyblue')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Feature Selection Methods Comparison')
    # Fix the rotation
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, horizontalalignment='right')

    # Add MSE values on top of bars
    for bar, mse in zip(bars, mses):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{mse:.4f}',
            ha='center',
            va='bottom'
        )

    # Feature Type Breakdown Plot (bottom)
    x = np.arange(len(methods))
    width = 0.25

    informative_counts = [analysis['n_informative'] for analysis in analyses]
    multicollinear_counts = [analysis['n_multicollinear'] for analysis in analyses]
    noise_counts = [analysis['n_noise'] for analysis in analyses]

    bar1 = ax2.bar(x - width, informative_counts, width, label='Informative', color='green', alpha=0.7)
    bar2 = ax2.bar(x, multicollinear_counts, width, label='Multicollinear', color='orange', alpha=0.7)
    bar3 = ax2.bar(x + width, noise_counts, width, label='Noise', color='red', alpha=0.7)

    # Add value labels on the bars
    def add_values(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if bar has height
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')

    add_values(bar1)
    add_values(bar2)
    add_values(bar3)

    ax2.set_ylabel('Number of Features Selected')
    ax2.set_title('Feature Type Breakdown by Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, horizontalalignment='right')
    ax2.legend()

    # Add a tight layout to prevent overlapping
    plt.tight_layout()
    
    # Save with extra padding to prevent cutoff
    plt.savefig('feature_selection_comparison_synthetic.png', 
                bbox_inches='tight', 
                dpi=300, 
                pad_inches=0.2)
    plt.close()

def main():
    SEED = 42
    set_all_seeds(SEED)
        
    # Device setup
    DEVICE_COUNT = torch.cuda.device_count()
    DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE_COUNT > 0 else 'CPU'
    device = torch.device('cuda:0' if DEVICE_COUNT > 0 else 'cpu')
    
    if device.type == 'cuda':
        print(f"Using GPU: {DEVICE_NAME}")
    else:
        print("Using CPU")
        
    # Create synthetic dataset with explicitly typed parameters
    n_samples = 1000
    n_features = 100
    n_informative = 1
    n_multicollinear = 1
    correlation_strength = 0.4
    noise_level = 1.5
    multicollinear_noise = 0.4

    n_select = 15
    n_episodes = 4000
    batch_size = 6
        
    X_np, y_np, feature_types = create_synthetic_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_multicollinear=n_multicollinear,
        correlation_strength=correlation_strength,
        noise_level=noise_level,
        multicollinear_noise=multicollinear_noise,
        random_state=SEED
    )

    # Print dataset information
    print("\nCreated synthetic dataset with:")
    print(f"- {n_features} total features")
    print(f"- {n_informative} informative features")
    print(f"- {n_multicollinear} multicollinear features per informative feature")
    print("\nFeature types:")
    for ftype, indices in feature_types.items():
        print(f"{ftype}: {sorted(indices)}")

    # Calculate and print correlations
    correlations = np.corrcoef(X_np.T)
    y_correlations = np.corrcoef(X_np.T, y_np)[:-1, -1]
    
    print("\nCorrelations with target:")
    for idx in feature_types['informative']:
        print(f"Informative feature {idx}: {y_correlations[idx]:.3f}")
    
    print("\nExample multicollinear relationships:")
    for inf_idx in feature_types['informative']:
        print(f"\nCorrelations with informative feature {inf_idx}:")
        related_indices = [i for i in feature_types['multicollinear'] 
                         if correlations[inf_idx, i] > 0.5]
        for rel_idx in related_indices:
            print(f"Feature {rel_idx}: {correlations[inf_idx, rel_idx]:.3f}")

    # Convert to torch tensors
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)

    # Initialize GFlowNet
    model = ImprovedGFlowNet(
        num_elements=n_features,
        target_size=n_select,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        device=device
    )

    # Initialize scheduler
    dataset_size = batch_size * n_episodes
    model.initialize_scheduler(dataset_size, batch_size, n_episodes)
    
    # Training loop
    print(f"\nTraining GFlowNet on {device}...")
    
    start_time = time.time()
    best_mse = float('inf')
    best_subset = None
    
    for episode in range(n_episodes):
        temp = max(1.0 - episode / n_episodes, 0.1)
        
        trajectories = []
        feature_subsets = []
        
        for _ in range(batch_size):
            subset, _ = model.sample_subset(temperature=temp)
            trajectories.append(subset)
            feature_subsets.append(subset)
        
        mses = evaluate_feature_subsets_concurrent(X, y, feature_subsets)
        mses = torch.tensor(mses, device=device)
        
        rewards = []
        for idx, (subset, mse) in enumerate(zip(trajectories, mses)):
            reward = -mse.item()
            rewards.append(reward)
            model.update_feature_stats(subset, reward)
            if mse.item() < best_mse:
                best_mse = mse.item()
                best_subset = subset.copy()
                print(f"New best reward (lowest MSE): {best_mse:.4f}")
        
        loss = model.train_step(trajectories, rewards, temp)
        
        if episode % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}, Loss: {loss:.4f}, "
                  f"Best MSE: {best_mse:.4f}, "
                  f"Temp: {temp:.4f}, Time: {elapsed:.1f}s")
    
    model.load_best_model()
    
    # Initialize results dictionary
    results = {}
    
    # Final evaluation using the actual best subset found during training
    print("\nFinal evaluation using best model (GFlowNet):")
    print(f"Best subset from training: {sorted(best_subset)}, MSE: {best_mse:.4f}")
    results['GFlowNet Best'] = {'features': best_subset, 'mse': best_mse}

    # Sample additional subsets for average performance
    final_subsets = []
    final_mses = []
    for _ in range(10):
        subset, _ = model.sample_subset(temperature=0.1)
        mse = evaluate_single_subset(X_np, y_np, subset)
        final_subsets.append(subset)
        final_mses.append(mse)
    
    avg_mse_gflownet = np.mean(final_mses)
    std_mse_gflownet = np.std(final_mses)
    results['GFlowNet Average'] = {'features': final_subsets[-1], 'mse': avg_mse_gflownet}
    
    # Feature importance analysis
    feature_importance = model.get_feature_importance()
    top_features_gflownet = np.argsort(feature_importance)[::-1][:n_select]
    
    print("\nTop features by importance (GFlowNet):")
    for idx, feature in enumerate(top_features_gflownet):
        print(f"Feature {feature}: {feature_importance[feature]:.4f}")
        if feature in feature_types['informative']:
            print("  (Informative feature)")
        elif feature in feature_types['multicollinear']:
            print("  (Multicollinear feature)")
    
    # Other feature selection methods
    # SelectKBest with f_regression
    selector = SelectKBest(score_func=f_regression, k=n_select)
    selector.fit(X_np, y_np)
    selected_features = np.where(selector.get_support())[0]
    mse = evaluate_single_subset(X_np, y_np, selected_features)
    results['SelectKBest (f_regression)'] = {'features': selected_features, 'mse': mse}
    print(f"\nSelectKBest (f_regression) - MSE: {mse:.4f}")
    
    # SelectKBest with mutual_info_regression
    selector = SelectKBest(score_func=mutual_info_regression, k=n_select)
    selector.fit(X_np, y_np)
    selected_features = np.where(selector.get_support())[0]
    mse = evaluate_single_subset(X_np, y_np, selected_features)
    results['SelectKBest (mutual_info)'] = {'features': selected_features, 'mse': mse}
    print(f"SelectKBest (mutual_info) - MSE: {mse:.4f}")
    
    # Random Forest Importance
    model_rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    model_rf.fit(X_np, y_np)
    importances = model_rf.feature_importances_
    selected_features = np.argsort(importances)[::-1][:n_select]
    mse = evaluate_single_subset(X_np, y_np, selected_features)
    results['Random Forest'] = {'features': selected_features, 'mse': mse}
    print(f"Random Forest Importance - MSE: {mse:.4f}")
    
    # Sequential Feature Selection
    selected_features_seq, mse_seq = sequential_feature_selection(X_np, y_np, n_select)
    results['Sequential Selection'] = {'features': selected_features_seq, 'mse': mse_seq}
    print(f"Sequential Selection - MSE: {mse_seq:.4f}")
    
    # Print comparison with detailed feature analysis
    print("\n=== Feature Selection Methods Comparison ===")
    for method, data in results.items():
        features = data['features']
        mse = data['mse']
        analysis = analyze_feature_selection(features, feature_types)
        
        print(f"\n{method}:")
        print(f"MSE: {mse:.4f}")
        print(f"Feature breakdown:")
        print(f"- Informative: {analysis['n_informative']}/{len(feature_types['informative'])}")
        print(f"- Multicollinear: {analysis['n_multicollinear']}/{len(feature_types['multicollinear'])}")
        print(f"- Noise: {analysis['n_noise']}/{len(feature_types['noise'])}")
        print(f"Selected features: {sorted(features)}")

    visualize_results(results, feature_types)


if __name__ == "__main__":
    main()