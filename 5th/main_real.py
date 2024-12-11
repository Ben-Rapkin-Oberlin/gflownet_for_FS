import torch
import numpy as np
import time
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

from gflownet4 import ImprovedGFlowNet
from utils import (
    set_all_seeds, evaluate_single_subset, evaluate_feature_subsets_concurrent,
    sequential_feature_selection, analyze_feature_selection, visualize_results2,
    pearson_feature_selection, distance_correlation
)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

VISUALIZATION_CONFIG = {
    'GFlowNet Best': True,
    'GFlowNet Average': False,
    'SelectKBest (f_regression)': True,
    'SelectKBest (mutual_info)': True,
    'Random Forest': True,
    'Sequential Selection': True,
    'Pearson Correlation': True,
    'Distance Correlation': False
}

def load_and_preprocess_data(data_path, target_column, test_size=0.2, random_state=41):
    """
    Load and preprocess the real dataset.
    
    Args:
        data_path (str): Path to the CSV file
        target_column (str): Name of the target variable column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train_np, X_test_np, y_train_np, y_test_np, feature_names, feature_types)
    """
    # Load the data
    df = pd.read_csv(data_path)#,nrows=1000)
    df = df.drop(columns=['CONSPUMA', 'CPUMA0010','APPAL','APPALD'])

    #print(df.columns.get_loc("DEPARTS"))
    #print(df.columns.get_loc("ARRIVES"))

    #exit()
    df = df.drop(df.iloc[:, 300:],axis = 1)
    if target_column == 'DEPARTS':
        print ('Departs True')

    df = df.drop(df[df['DEPARTS'] == 0].index)#, inplace = True)
    drop_indices = np.random.choice(df.index, 500000, replace=False)
    df = df.drop(drop_indices)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Create a simple feature_types dictionary
    # In real data, we don't know the true feature types, so we'll just track selected features
    feature_types = {
        'selected': [],
        'not_selected': list(range(len(feature_names)))
    }
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale the target variable
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
            feature_names, feature_types)

def main():
    # Configuration
    SEED = 41
    set_all_seeds(SEED)
    
    # Data loading configuration
    DATA_PATH = "encoded_departs.csv"  # Replace with your dataset path
    TARGET_COLUMN = "DEPARTS"     # Replace with your target column name
    n_select = 10  # Number of features to select
    n_episodes = 2000
    batch_size = 2
    
    # Device setup
    DEVICE_COUNT = torch.cuda.device_count()
    DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE_COUNT > 0 else 'CPU'
    device = torch.device('cuda:0' if DEVICE_COUNT > 0 else 'cpu')
    
    if device.type == 'cuda':
        print(f"Using GPU: {DEVICE_NAME}")
    else:
        print("Using CPU")

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    (X_train_np, X_test_np, y_train_np, y_test_np, 
     feature_names, feature_types) = load_and_preprocess_data(
        DATA_PATH, TARGET_COLUMN
    )
    
    n_features = X_train_np.shape[1]
    print(f"\nDataset loaded successfully:")
    print(f"Training samples: {X_train_np.shape[0]}")
    print(f"Testing samples: {X_test_np.shape[0]}")
    print(f"Number of features: {n_features}")
    
    # Convert to torch tensors for training
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
    
    # Initialize GFlowNet
    model = ImprovedGFlowNet(
        num_elements=n_features,
        target_size=n_select,
        hidden_dim=320,
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
        
        trajectories = []
        feature_subsets = []
        
        for _ in range(batch_size):
            subset, _ = model.sample_subset(temperature=temp)
            trajectories.append(subset)
            feature_subsets.append(subset)
        
        mses = evaluate_feature_subsets_concurrent(X_train, y_train, feature_subsets)
        mses = torch.tensor(mses, device=device)
        
        rewards = []
        for subset, mse in zip(trajectories, mses):
            reward = -mse.item()
            rewards.append(reward)
            model.update_feature_stats(subset, reward)
            if mse.item() < best_mse:
                best_mse = mse.item()
                best_subset = subset.copy()
                print(f"New best reward (lowest MSE): {best_mse:.4f}")
        
        loss = model.train_step(trajectories, rewards, temp)
        
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}, Loss: {loss:.4f}, "
                  f"Best MSE: {best_mse:.4f}, "
                  f"Temp: {temp:.4f}, Time: {elapsed:.1f}s")
    
    # Results dictionary
    results = {}
    
    # Final evaluation using the best subset
    print("\nFinal evaluation using best model (GFlowNet):")
    print("Best subset features:")
    for idx in sorted(best_subset):
        print(f"  {idx}: {feature_names[idx]}")
    print(f"MSE: {best_mse:.4f}")
    results['GFlowNet Best'] = {'features': best_subset, 'mse': best_mse}
    
    # Sample additional subsets for average performance
    final_subsets = []
    final_mses = []
    for _ in range(10):
        subset, _ = model.sample_subset(temperature=0.1)
        mse = evaluate_single_subset(X_train_np, y_train_np, subset)
        final_subsets.append(subset)
        final_mses.append(mse)
    
    avg_mse_gflownet = np.mean(final_mses)
    results['GFlowNet Average'] = {'features': final_subsets[-1], 'mse': avg_mse_gflownet}
    
    # Feature importance analysis
    feature_importance = model.get_feature_importance()
    top_features_gflownet = np.argsort(feature_importance)[::-1][:n_select]
    
    print("\nTop features by importance (GFlowNet):")
    for idx, feature in enumerate(top_features_gflownet):
        print(f"{feature_names[feature]}: {feature_importance[feature]:.4f}")
    
    # Other feature selection methods
    # SelectKBest with f_regression
    print('calculating f-reg')
    a=time.time()
    selector = SelectKBest(score_func=f_regression, k=n_select)
    selector.fit(X_train_np, y_train_np)
    selected_features = np.where(selector.get_support())[0]
    mse = evaluate_single_subset(X_train_np, y_train_np, selected_features)
    results['SelectKBest (f_regression)'] = {'features': selected_features, 'mse': mse}
    print(time.time()-a)

    # SelectKBest with mutual_info_regression
    print('calculating MI')
    a=time.time()
    selector = SelectKBest(score_func=mutual_info_regression, k=n_select)
    selector.fit(X_train_np, y_train_np)
    selected_features = np.where(selector.get_support())[0]
    mse = evaluate_single_subset(X_train_np, y_train_np, selected_features)
    results['SelectKBest (mutual_info)'] = {'features': selected_features, 'mse': mse}
    print(time.time()-a)

    # Random Forest Importance
    print('calculating RF')
    a=time.time()
    model_rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    model_rf.fit(X_train_np, y_train_np)
    importances = model_rf.feature_importances_
    selected_features = np.argsort(importances)[::-1][:n_select]
    mse = evaluate_single_subset(X_train_np, y_train_np, selected_features)
    results['Random Forest'] = {'features': selected_features, 'mse': mse}
    print(time.time()-a)

    # Sequential Feature Selection
    print('calculating SFS')
    a=time.time()
    selected_features_seq, mse_seq = sequential_feature_selection(
        X_train_np, y_train_np, n_select
    )
    results['Sequential Selection'] = {'features': selected_features_seq, 'mse': mse_seq}
    print(time.time()-a)

    # Pearson correlation selection
    print('calculating P')
    a=time.time()
    selected_features_pearson = pearson_feature_selection(X_train_np, y_train_np, n_select)
    mse_pearson = evaluate_single_subset(X_train_np, y_train_np, selected_features_pearson)
    results['Pearson Correlation'] = {
        'features': selected_features_pearson, 
        'mse': mse_pearson
    }
    print(time.time()-a)

    # Distance correlation selection
    #print('calculating DC')
    #a=time.time()
    #dcor = distance_correlation(X_train_np, y_train_np)
    #selected_features_dcor = np.argsort(dcor)[::-1][:n_select]
    #mse_dcor = evaluate_single_subset(X_train_np, y_train_np, selected_features_dcor)
    #results['Distance Correlation'] = {'features': selected_features_dcor, 'mse': mse_dcor}
    #print(time.time()-a)

    # Print comparison
    print("\n=== Feature Selection Methods Comparison ===")
    for method, data in results.items():
        if VISUALIZATION_CONFIG.get(method, False):
            features = data['features']
            mse = data['mse']
            print(f"\n{method}:")
            print(f"MSE: {mse:.4f}")
            print("Selected features:")
            for idx in sorted(features):
                print(f"  {idx}: {feature_names[idx]}")

    # Create visualization
    visualize_results2(results, feature_names, str(time.time()),n_episodes)


if __name__ == "__main__":
    main()