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
from gflownet4 import ImprovedGFlowNet
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from scipy.spatial.distance import pdist, squareform
import numpy as np
warnings.filterwarnings('ignore')




from create_data import create_synthetic_dataset




def distance_correlation(X, y):
    """
    Calculate the distance correlation between features and target.
    Returns a vector of distance correlations for each feature.
    """
    n = len(y)
    
    # Calculate distance matrices
    def compute_distance_matrix(data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        D = squareform(pdist(data))
        return D
    
    # Center distance matrices
    def center_distance_matrix(D):
        means_rows = D.mean(axis=0)
        means_cols = D.mean(axis=1)
        mean_all = D.mean()
        return D - means_rows - means_cols[:, np.newaxis] + mean_all
    
    # Calculate distance correlations for each feature
    dcor = np.zeros(X.shape[1])
    y_dist = compute_distance_matrix(y.reshape(-1, 1))
    y_centered = center_distance_matrix(y_dist)
    
    for i in range(X.shape[1]):
        X_dist = compute_distance_matrix(X[:, i].reshape(-1, 1))
        X_centered = center_distance_matrix(X_dist)
        
        # Calculate distance correlation
        numerator = np.sqrt(np.sum(X_centered * y_centered))
        denominator = np.sqrt(np.sum(X_centered * X_centered) * np.sum(y_centered * y_centered))
        
        if denominator > 0:
            dcor[i] = numerator / denominator
        else:
            dcor[i] = 0
            
    return dcor

def pearson_feature_selection(X, y, n_select):
    """
    Select features based on absolute Pearson correlation with target.
    """
    correlations = np.array([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    selected_features = np.argsort(np.abs(correlations))[::-1][:n_select]
    return selected_features

def visualize_results(results, feature_types):
    """Create visualization of results based on configuration."""
    # Filter results based on configuration
    filtered_results = {k: v for k, v in results.items() if VISUALIZATION_CONFIG.get(k, False)}
    
    # Get method names and their metrics
    methods = list(filtered_results.keys())
    mses = [ensure_cpu(data['mse']) for data in filtered_results.values()]
    analyses = [analyze_feature_selection(data['features'], feature_types) 
               for data in filtered_results.values()]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[1, 1.2])

    # MSE Plot (top)
    bars = ax1.bar(methods, mses, color='skyblue')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Feature Selection Methods Comparison')
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
            if height > 0:
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
    a=str(time.time())

    plt.tight_layout()
    plt.savefig('clean_5000_feature_selection_comparison_synthetic_'+a+'.png', 
                bbox_inches='tight', 
                dpi=300,
                pad_inches=0.2)
    plt.close()

def set_all_seeds(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# Configuration for methods to include in visualization
VISUALIZATION_CONFIG = {
    'GFlowNet Best': True,
    'GFlowNet Average': False,
    'SelectKBest (f_regression)': True,
    'SelectKBest (mutual_info)': True,
    'Random Forest': True,
    'Sequential Selection': True,
    'Pearson Correlation': True,
    'Distance Correlation': True
}

def main():
    SEED = 41
    set_all_seeds(SEED)

    n_select = 10
    n_episodes = 5000
    batch_size = 6   
    n_features = 100
 
    # Device setup
    DEVICE_COUNT = torch.cuda.device_count()
    DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE_COUNT > 0 else 'CPU'
    device = torch.device('cuda:0' if DEVICE_COUNT > 0 else 'cpu')
    
    if device.type == 'cuda':
        print(f"Using GPU: {DEVICE_NAME}")
    else:
        print("Using CPU")
    

    dataset_config = {
        'n_samples': 1000,
        'n_features': n_features,
        'n_informative': 3,
        'n_multicollinear': 2,
        'correlation_strength': 0.9,    # Very strong correlations
        'noise_level': 0.1,            # Minimal noise
        'multicollinear_noise': 0.05,  # Very clean multicollinear relationships
        'feature_noise': 0.1,
        'noise_config': {
            'global_noise_scale': 0.3,      # Low overall noise
            'feature_noise_scale': 0.2,     # Clean features
            'target_noise_scale': 0.2,      # Clean target
            'informative_noise': 0.1,       # Very clear informative features
            'noise_feature_std': 0.2,       # Low noise feature variance
            'signal_to_noise_ratio': 20.0   # Very high SNR
        },
        'nonlinear_features': None,         # Only linear relationships
        'interaction_features': None,        # No interactions
        'feature_distributions': {
            0: 'normal',                    # All normal distributions
            1: 'normal',
            2: 'normal'
        },
        'outlier_config': None,             # No outliers
        'heteroscedastic_noise': None       # Homoscedastic noise only
    }



    print("\n=== Dataset Configuration ===")
    print(f"Total samples: {dataset_config['n_samples']}")
    print(f"Total features: {dataset_config['n_features']}")
    print(f"Informative features: {dataset_config['n_informative']}")
    print(f"Multicollinear features per informative: {dataset_config['n_multicollinear']}")
    
    # Create dataset
    X_np, y_np, feature_types = create_synthetic_dataset(**dataset_config)
    

    from data_info import disc_data
    disc_data(dataset_config,X_np, y_np, feature_types)

    # Convert to torch tensors and continue with model training
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)


    # Initialize GFlowNet
    model = ImprovedGFlowNet(
        num_elements=n_features,
        target_size=n_select,
        hidden_dim=256,
        num_heads=4,
        num_layers=3,
        device=device
    )
    
    # Training loop
    print(f"\nTraining GFlowNet on {device}...")
    
    start_time = time.time()
    best_mse = float('inf')
    best_subset = None
    
    for episode in range(n_episodes):
        temp = max(8.0 - (8*episode) / n_episodes, 0.3)
        #temperature = max(min(temperature, 5.0), 0.3)


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
    #model.load_best_model()
    
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
    
    # Add Pearson correlation selection
    selected_features_pearson = pearson_feature_selection(X_np, y_np, n_select)
    mse_pearson = evaluate_single_subset(X_np, y_np, selected_features_pearson)
    results['Pearson Correlation'] = {'features': selected_features_pearson, 'mse': mse_pearson}
    print(f"Pearson Correlation - MSE: {mse_pearson:.4f}")
    
    # Add Distance correlation selection
    dcor = distance_correlation(X_np, y_np)
    selected_features_dcor = np.argsort(dcor)[::-1][:n_select]
    mse_dcor = evaluate_single_subset(X_np, y_np, selected_features_dcor)
    results['Distance Correlation'] = {'features': selected_features_dcor, 'mse': mse_dcor}
    print(f"Distance Correlation - MSE: {mse_dcor:.4f}")
    
    # Other feature selection methods remain the same...
    
    # Print comparison with detailed feature analysis
    print("\n=== Feature Selection Methods Comparison ===")
    for method, data in results.items():
        # Only print methods that are configured to be shown
        if VISUALIZATION_CONFIG.get(method, False):
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

    # Create visualization based on configuration
    visualize_results(results, feature_types)


if __name__ == "__main__":
    main()