import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import pdist, squareform


def distance_correlation(X, y):
    """
    Calculate the distance correlation between features and target.
    Returns a vector of distance correlations for each feature.
    """
    n = len(y)
    
    def compute_distance_matrix(data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        D = squareform(pdist(data))
        return D
    
    def center_distance_matrix(D):
        means_rows = D.mean(axis=0)
        means_cols = D.mean(axis=1)
        mean_all = D.mean()
        return D - means_rows - means_cols[:, np.newaxis] + mean_all
    
    dcor = np.zeros(X.shape[1])
    y_dist = compute_distance_matrix(y.reshape(-1, 1))
    y_centered = center_distance_matrix(y_dist)
    
    for i in range(X.shape[1]):
        X_dist = compute_distance_matrix(X[:, i].reshape(-1, 1))
        X_centered = center_distance_matrix(X_dist)
        
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

def visualize_results(results, feature_types, timestamp):
    """Create visualization of results based on configuration."""
    filtered_results = {k: v for k, v in results.items() if VISUALIZATION_CONFIG.get(k, False)}
    
    methods = list(filtered_results.keys())
    mses = [ensure_cpu(data['mse']) for data in filtered_results.values()]
    analyses = [analyze_feature_selection(data['features'], feature_types) 
               for data in filtered_results.values()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[1, 1.2])

    # MSE Plot
    bars = ax1.bar(methods, mses, color='skyblue')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Feature Selection Methods Comparison')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, horizontalalignment='right')

    for bar, mse in zip(bars, mses):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{mse:.4f}',
            ha='center',
            va='bottom'
        )

    # Feature Type Breakdown Plot
    x = np.arange(len(methods))
    width = 0.25

    informative_counts = [analysis['n_informative'] for analysis in analyses]
    multicollinear_counts = [analysis['n_multicollinear'] for analysis in analyses]
    noise_counts = [analysis['n_noise'] for analysis in analyses]

    bar1 = ax2.bar(x - width, informative_counts, width, label='Informative', color='green', alpha=0.7)
    bar2 = ax2.bar(x, multicollinear_counts, width, label='Multicollinear', color='orange', alpha=0.7)
    bar3 = ax2.bar(x + width, noise_counts, width, label='Noise', color='red', alpha=0.7)

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

    plt.tight_layout()
    plt.savefig(f'TEST_feature_selection_comparison_synthetic_{timestamp}.png', 
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

def sequential_feature_selection(X, y, n_features, method='forward'):
    """
    Performs sequential feature selection with improved handling of singular matrices.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        n_features (int): Number of features to select
        method (str): 'forward' or 'backward' selection
        
    Returns:
        tuple: (selected features indices, final MSE score)
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    
    n_total_features = X.shape[1]
    if method == 'forward':
        selected_features = []
        remaining_features = list(range(n_total_features))
    else:
        selected_features = list(range(n_total_features))
        remaining_features = []
    
    # Initialize Ridge regression with a small alpha for numerical stability
    model = Ridge(alpha=1e-3)
    
    best_mse = float('inf')
    
    while len(selected_features) < n_features if method == 'forward' else len(selected_features) > n_features:
        best_new_mse = float('inf')
        best_feature = None
        
        # Try adding (or removing) each remaining feature
        for feature in (remaining_features if method == 'forward' else selected_features):
            # Create new feature set
            if method == 'forward':
                current_features = selected_features + [feature]
            else:
                current_features = [f for f in selected_features if f != feature]
            
            # Skip if we don't have enough features to make a prediction
            if len(current_features) == 0:
                continue
                
            # Evaluate the feature set
            try:
                X_subset = X[:, current_features]
                model.fit(X_subset, y)
                y_pred = model.predict(X_subset)
                mse = mean_squared_error(y, y_pred)
                
                if mse < best_new_mse:
                    best_new_mse = mse
                    best_feature = feature
                    
            except Exception as e:
                continue
        
        # Update features lists
        if best_feature is not None:
            if method == 'forward':
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
            else:
                selected_features.remove(best_feature)
                remaining_features.append(best_feature)
            best_mse = best_new_mse
        else:
            # If no improvement found, break early
            break
            
    return selected_features, best_mse
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


def visualize_results2(results, feature_names, timestamp):
    """
    Visualize feature selection results for real data.
    
    Args:
        results (dict): Dictionary containing results from different methods
        feature_names (list): List of feature names
        timestamp (str): Timestamp for saving the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from collections import Counter
    
    # Set up the plotting style
    #plt.style.use('seaborn')
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. MSE Comparison (Subplot 1)
    plt.subplot(2, 1, 1)
    methods = []
    mses = []
    for method, data in results.items():
        if VISUALIZATION_CONFIG.get(method, False):
            methods.append(method)
            mses.append(data['mse'])
    
    # Create bar plot for MSEs
    bars = plt.bar(methods, mses)
    plt.xticks(rotation=45, ha='right')
    plt.title('MSE Comparison Across Methods')
    plt.ylabel('Mean Squared Error')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # 2. Feature Overlap Analysis (Subplot 2)
    plt.subplot(2, 1, 2)
    
    # Count feature occurrence across methods
    all_selected_features = []
    for method, data in results.items():
        if VISUALIZATION_CONFIG.get(method, False):
            all_selected_features.extend(data['features'])
    
    feature_counts = Counter(all_selected_features)
    most_common_features = feature_counts.most_common(10)  # Top 10 most commonly selected features
    
    # Create bar plot for feature frequency
    feature_names_short = [f"{feature_names[feat]} ({feat})" 
                         for feat, _ in most_common_features]
    frequencies = [count for _, count in most_common_features]
    
    bars = plt.bar(feature_names_short, frequencies)
    plt.xticks(rotation=45, ha='right')
    plt.title('Most Frequently Selected Features Across Methods')
    plt.ylabel('Number of Methods Selecting Feature')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'feature_selection_comparison_{timestamp}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

    # Create a detailed results table
    create_results_table(results, feature_names, timestamp)

def create_results_table(results, feature_names, timestamp):
    """
    Create a detailed results table showing selected features and their frequencies.
    
    Args:
        results (dict): Dictionary containing results from different methods
        feature_names (list): List of feature names
        timestamp (str): Timestamp for saving the table
    """
    import pandas as pd
    
    # Create a DataFrame to store feature selection frequencies
    all_features = set()
    for method, data in results.items():
        if VISUALIZATION_CONFIG.get(method, False):
            all_features.update(data['features'])
    
    # Initialize the comparison table
    comparison_table = pd.DataFrame(index=sorted(all_features))
    
    # Fill in the table
    for method, data in results.items():
        if VISUALIZATION_CONFIG.get(method, False):
            comparison_table[method] = [1 if i in data['features'] else 0 
                                     for i in comparison_table.index]
    
    # Add feature names
    comparison_table['Feature Name'] = [feature_names[i] for i in comparison_table.index]
    
    # Add total count
    comparison_table['Times Selected'] = comparison_table.drop('Feature Name', axis=1).sum(axis=1)
    
    # Reorder columns
    cols = ['Feature Name', 'Times Selected'] + [col for col in comparison_table.columns 
                                               if col not in ['Feature Name', 'Times Selected']]
    comparison_table = comparison_table[cols]
    
    # Sort by frequency
    comparison_table = comparison_table.sort_values('Times Selected', ascending=False)
    
    # Save to CSV
    comparison_table.to_csv(f'feature_selection_comparison_{timestamp}.csv')

    return comparison_table




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