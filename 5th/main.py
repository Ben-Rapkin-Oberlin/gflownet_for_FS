import torch
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from create_data import create_synthetic_dataset
from gflownet4 import ImprovedGFlowNet
from data_info import disc_data
from utils import (
    set_all_seeds, evaluate_single_subset, evaluate_feature_subsets_concurrent,
    sequential_feature_selection, analyze_feature_selection, visualize_results,
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
    'Distance Correlation': True
}####################################################must change in both rn

def main():
    SEED = 41
    set_all_seeds(SEED)

    n_select = 10
    n_episodes = 10
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

    # Dataset configuration
    dataset_config = {
        'n_samples': 1000,
        'n_features': n_features,
        'n_informative': 3,
        'n_multicollinear': 2,
        'correlation_strength': 0.9,
        'noise_level': 0.1,
        'multicollinear_noise': 0.05,
        'feature_noise': 0.1,
        'noise_config': {
            'global_noise_scale': 0.3,
            'feature_noise_scale': 0.2,
            'target_noise_scale': 0.2,
            'informative_noise': 0.1,
            'noise_feature_std': 0.2,
            'signal_to_noise_ratio': 20.0
        },
        'nonlinear_features': None,
        'interaction_features': None,
        'feature_distributions': {
            0: 'normal',
            1: 'normal',
            2: 'normal'
        },
        'outlier_config': None,
        'heteroscedastic_noise': None
    }

    print("\n=== Dataset Configuration ===")
    print(f"Total samples: {dataset_config['n_samples']}")
    print(f"Total features: {dataset_config['n_features']}")
    print(f"Informative features: {dataset_config['n_informative']}")
    print(f"Multicollinear features per informative: {dataset_config['n_multicollinear']}")
    
    # Create dataset
    X_np, y_np, feature_types = create_synthetic_dataset(**dataset_config)
    disc_data(dataset_config, X_np, y_np, feature_types)

    # Convert to torch tensors
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
    
    # Print comparison with detailed feature analysis
    print("\n=== Feature Selection Methods Comparison ===")
    for method, data in results.items():
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

    # Create visualization
    visualize_results(results, feature_types, str(time.time()))

if __name__ == "__main__":
    main()