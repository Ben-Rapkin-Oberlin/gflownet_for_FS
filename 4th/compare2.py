# compare2.py

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_all_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Your GFlowNet implementation (assuming it's in gflownet3.py and contains ImprovedGFlowNet class)
# Make sure to import your own GFlowNet implementation
from gflownet3 import ImprovedGFlowNet

# Data loading and preprocessing
def load_and_clean_data():
    """
    Load and clean the communities_and_crime dataset.
    Returns:
        X (pd.DataFrame): Cleaned feature set.
        y (pd.Series): Target variable.
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
    """
    # Fetch dataset
    communities_and_crime = fetch_ucirepo(id=183)
    
    # Extract features and target
    X = communities_and_crime.data.features.copy()
    y = communities_and_crime.data.targets.copy()
    
    # Replace '?' with NaN
    X.replace('?', np.nan, inplace=True)
    
    # Drop irrelevant columns
    columns_to_drop = ['state', 'county', 'community', 'communityname', 'fold']
    X = X.drop(columns=columns_to_drop)
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Impute numerical columns with median
    from sklearn.impute import SimpleImputer
    num_imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    # Impute categorical columns with most frequent
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # Convert data types
    X[numerical_cols] = X[numerical_cols].astype(float)
    X[categorical_cols] = X[categorical_cols].astype(str)
    
    return X, y, numerical_cols, categorical_cols

def one_hot_encode_categoricals(X, categorical_cols):
    """
    One-hot encode categorical columns.
    Returns:
        X (pd.DataFrame): DataFrame with encoded features.
        encoded_cols (list): List of encoded column names.
    """
    if not categorical_cols:
        return X, []
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cats = encoder.fit_transform(X[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cols, index=X.index)
    X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
    return X, encoded_cols

def reduce_categorical_features(X, encoded_cols, reduction_ratio=0.5):
    """
    Remove a portion of one-hot encoded features based on variance.
    Returns:
        X_reduced (pd.DataFrame): DataFrame with reduced features.
    """
    if len(encoded_cols) == 0:
        return X
    # Calculate variance of one-hot encoded features
    variances = X[encoded_cols].var()
    # Determine the number of features to remove
    num_features_to_remove = int(len(encoded_cols) * reduction_ratio)
    # Get features with the lowest variance
    features_to_remove = variances.nsmallest(num_features_to_remove).index.tolist()
    # Drop the selected features
    X_reduced = X.drop(columns=features_to_remove)
    print(f"Removed {len(features_to_remove)} categorical features based on lowest variance.")
    return X_reduced

def prepare_dataset():
    X, y, numerical_cols, categorical_cols = load_and_clean_data()
    print(f"Initial number of features: {X.shape[1]}")
    
    # Ensure y is a 1-dimensional Series
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            raise ValueError("Target variable 'y' has more than one column.")
    elif isinstance(y, np.ndarray):
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze()
        elif y.ndim > 1:
            raise ValueError("Target variable 'y' has more than one dimension.")
    
    # One-hot encode categorical variables
    if categorical_cols:
        X, encoded_cols = one_hot_encode_categoricals(X, categorical_cols)
        print(f"Number of features after one-hot encoding: {X.shape[1]}")
        
        # Reduce the number of one-hot encoded features
        X = reduce_categorical_features(X, encoded_cols, reduction_ratio=0.5)
        print(f"Number of features after reducing categorical features: {X.shape[1]}")
    else:
        print("No categorical columns to encode.")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Final checks
    assert X.isnull().sum().sum() == 0, "There are still missing values."
    
    return X, y

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
            try:
                mse = future.result()
                mses.append(mse)
            except Exception as e:
                print(f"Error evaluating subset {future_to_subset[future]}: {e}")
    return mses

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
    
    # Load and prepare data
    X_df, y_series = prepare_dataset()
    print(f"Dataset loaded with {X_df.shape[0]} samples and {X_df.shape[1]} features.")
    
    # Convert to NumPy arrays
    X_np = X_df.values
    y_np = y_series.values  # Ensure y_series is 1-dimensional
    
    # Convert to Torch tensors
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    
    n_samples, n_features = X_np.shape
    n_select = 10  # Number of features to select
    n_episodes = 2
    batch_size = 1  # Adjusted based on your GPU memory
    
    # Initialize GFlowNet model
    model = ImprovedGFlowNet(
        num_elements=n_features,
        target_size=n_select,
        hidden_dim=100,  # Adjusted hidden_dim to manage memory usage
        num_heads=2,    # Adjusted num_heads to reduce memory usage
        num_layers=2,   # Adjusted num_layers to reduce memory usage
        device=device
    )

    # Training loop
    print(f"Training GFlowNet on {device}...")
    
    start_time = time.time()
    best_reward = float('inf')
    best_subset = None

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
                best_subset = subset  # Save the best subset
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
    
    if best_subset is not None:
        mse = evaluate_single_subset(X_np, y_np, best_subset)
        #final_subsets.append(best_subset)
        #final_mses.append(mse)
        print(f"Best subset from training: {best_subset}, MSE: {mse:.4f}")

    # Sample additional subsets using the trained model
    for _ in range(10):
        subset, _ = model.sample_subset(temperature=0.05)
        mse = evaluate_single_subset(X_np, y_np, subset)
        final_subsets.append(subset)
        final_mses.append(mse)
    print(f"Sampled subset: {subset}, MSE: {mse:.4f}")
    
    # Calculate average MSE for GFlowNet
    avg_mse_gflownet = np.mean(final_mses)
    std_mse_gflownet = np.std(final_mses)
    print(f"\nAverage MSE (GFlowNet): {avg_mse_gflownet:.4f} ± {std_mse_gflownet:.4f}")
    
    # Feature importance analysis (top features)
    feature_counts = np.zeros(n_features)
    for subset in final_subsets:
        feature_counts[subset] += 1
    top_features_gflownet = np.argsort(feature_counts)[::-1][:n_select]
    print("\nTop features by selection frequency (GFlowNet):")
    for idx in top_features_gflownet:
        feature_name = X_df.columns[idx]
        print(f"Feature {idx} ({feature_name}): Selected {int(feature_counts[idx])} times")
    
    # Dictionary to store results
    results = {}
    
    # Define feature selection methods
    k = n_select  # Number of features to select
    
    # Convert Torch tensors back to DataFrame for other methods
    X_df = pd.DataFrame(X_np, columns=X_df.columns)
    y_series = pd.Series(y_np)
    
    # Test 1: Univariate Feature Selection with f_regression
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_df, y_series)
    selected_features = X_df.columns[selector.get_support()].tolist()
    feature_indices = [X_df.columns.get_loc(c) for c in selected_features]
    mse = evaluate_single_subset(X_np, y_np, feature_indices)
    results['SelectKBest (f_regression)'] = {'features': selected_features, 'mse': mse}
    print(f"\nSelectKBest (f_regression) - MSE: {mse:.4f}")
    
    # Test 2: Univariate Feature Selection with mutual_info_regression
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    selector.fit(X_df, y_series)
    selected_features = X_df.columns[selector.get_support()].tolist()
    feature_indices = [X_df.columns.get_loc(c) for c in selected_features]
    mse = evaluate_single_subset(X_np, y_np, feature_indices)
    results['SelectKBest (mutual_info_regression)'] = {'features': selected_features, 'mse': mse}
    print(f"SelectKBest (mutual_info_regression) - MSE: {mse:.4f}")
    
    # Test 4: Feature Importance from Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    model_rf.fit(X_df, y_series)
    importances = model_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = X_df.columns[indices[:k]].tolist()
    feature_indices = indices[:k].tolist()
    mse = evaluate_single_subset(X_np, y_np, feature_indices)
    results['Random Forest Importance'] = {'features': selected_features, 'mse': mse}
    print(f"Random Forest Importance - MSE: {mse:.4f}")
    
    # Test 6: Correlation-Based Selection
    correlation = X_df.corrwith(y_series).abs()
    selected_features = correlation.sort_values(ascending=False).head(k).index.tolist()
    feature_indices = [X_df.columns.get_loc(c) for c in selected_features]
    mse = evaluate_single_subset(X_np, y_np, feature_indices)
    results['Correlation-Based Selection'] = {'features': selected_features, 'mse': mse}
    print(f"Correlation-Based Selection - MSE: {mse:.4f}")
    
    # Add GFlowNet Average and Best to results

    # Convert best_subset indices to feature names
    if best_subset is not None:
        try:
            best_feature_names = [X_df.columns[idx] for idx in best_subset]
            results['GFlowNet Best'] = {'features': best_feature_names, 'mse': best_reward}
        except IndexError as e:
            print(f"Error converting best_subset indices to feature names: {e}")
            results['GFlowNet Best'] = {'features': [], 'mse': best_reward}
    else:
        results['GFlowNet Best'] = {'features': [], 'mse': best_reward}

    # Get top feature names from GFlowNet
    try:
        top_feature_names_gflownet = [X_df.columns[idx] for idx in top_features_gflownet]
        results['GFlowNet Average'] = {'features': top_feature_names_gflownet, 'mse': avg_mse_gflownet}
    except IndexError as e:
        print(f"Error converting top_features_gflownet indices to feature names: {e}")
        results['GFlowNet Average'] = {'features': [], 'mse': avg_mse_gflownet}

    # Comparison
    print("\n=== Feature Selection Methods Comparison ===")
    print(f"GFlowNet Average MSE: {avg_mse_gflownet:.4f} ± {std_mse_gflownet:.4f}")
    print(f"GFlowNet Best MSE: {best_reward:.4f}")
    for method, data in results.items():
        if method not in ['GFlowNet Average', 'GFlowNet Best']:
            print(f"{method}: MSE = {data['mse']:.4f}")

    # Identify overlapping features
    # Get GFlowNet features as a set of feature indices
    gflownet_features_set = set()
    for feature_name in top_feature_names_gflownet:
        try:
            idx = X_df.columns.get_loc(feature_name)
            gflownet_features_set.add(idx)
        except KeyError as e:
            print(f"Feature name '{feature_name}' not found in DataFrame columns: {e}")

    for method, data in results.items():
        # Skip methods where 'features' is not a list
        if not isinstance(data['features'], list):
            print(f"Skipping method '{method}' as 'features' is not a list.")
            continue  # Skip this method
        # Get feature indices for this method
        method_features_indices = []
        for c in data['features']:
            try:
                idx = X_df.columns.get_loc(c)
                method_features_indices.append(idx)
            except KeyError:
                print(f"Feature '{c}' from method '{method}' not found in DataFrame columns.")
        method_features_set = set(method_features_indices)
        common_features = gflownet_features_set & method_features_set
        print(f"\nCommon features between GFlowNet and {method}:")
        if common_features:
            for idx in common_features:
                feature_name = X_df.columns[idx]
                print(f"Feature {idx} ({feature_name})")
        else:
            print("No common features.")

    # Visualize the comparison
    methods = ['GFlowNet Average', 'GFlowNet Best'] + [m for m in results.keys() if m not in ['GFlowNet Average', 'GFlowNet Best']]
    mses = [
        results['GFlowNet Average']['mse'],
        results['GFlowNet Best']['mse']
    ] + [
        data['mse'] for m, data in results.items() if m not in ['GFlowNet Average', 'GFlowNet Best']
    ]

    plt.figure(figsize=(14, 8))
    bars = plt.bar(methods, mses, color='skyblue')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Feature Selection Methods Comparison (Top {k} Features)')
    plt.xticks(rotation=45, ha='right')

    # Annotate bars with MSE values
    for bar, mse in zip(bars, mses):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f'{mse:.4f}',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
