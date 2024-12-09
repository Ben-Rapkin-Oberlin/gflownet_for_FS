
import numpy as np
from scipy import stats

def disc_data(dataset_config,X_np, y_np, feature_types): 
    print("\n=== Dataset Configuration ===")
    print(f"Total samples: {dataset_config['n_samples']}")
    print(f"Total features: {dataset_config['n_features']}")
    print(f"Informative features: {dataset_config['n_informative']}")
    print(f"Multicollinear features per informative: {dataset_config['n_multicollinear']}")
    
    print("\n=== Dataset Characteristics ===")
    print("\nFeature Distributions:")
    if dataset_config.get('feature_distributions'):
        for feat_idx, dist_type in dataset_config['feature_distributions'].items():
            print(f"Feature {feat_idx}: {dist_type}")
            data = X_np[:, feat_idx]
            print(f"  - Mean: {np.mean(data):.3f}")
            print(f"  - Std: {np.std(data):.3f}")
            print(f"  - Skewness: {stats.skew(data):.3f}")
            print(f"  - Kurtosis: {stats.kurtosis(data):.3f}")
    else:
        print("All features follow normal distribution")
    
    print("\nNon-linear Relationships:")
    if dataset_config.get('nonlinear_features'):
        for rel_type, features in dataset_config['nonlinear_features'].items():
            print(f"{rel_type.capitalize()} relationships in features: {features}")
    else:
        print("No non-linear relationships")
    
    print("\nInteraction Effects:")
    if dataset_config.get('interaction_features'):
        for f1, f2 in dataset_config['interaction_features']:
            interaction_effect = np.corrcoef(X_np[:, f1] * X_np[:, f2], y_np)[0,1]
            print(f"Interaction between features {f1} and {f2}: {interaction_effect:.3f} correlation with target")
    else:
        print("No interaction effects")
    
    print("\nOutlier Information:")
    if dataset_config.get('outlier_config'):
        print(f"Contamination rate: {dataset_config['outlier_config']['contamination']}")
        print(f"Magnitude: {dataset_config['outlier_config']['magnitude']}x")
    else:
        print("No outliers in dataset")
    
    print("\nNoise Configuration:")
    noise_config = dataset_config.get('noise_config', {})
    print(f"Global noise scale: {noise_config.get('global_noise_scale', 'Not specified')}")
    print(f"Feature noise scale: {noise_config.get('feature_noise_scale', 'Not specified')}")
    print(f"Target noise scale: {noise_config.get('target_noise_scale', 'Not specified')}")
    print(f"Signal-to-noise ratio: {noise_config.get('signal_to_noise_ratio', 'Not specified')}")
    
    # Calculate and print correlations
    print("\n=== Correlation Analysis ===")
    correlations = np.corrcoef(X_np.T)
    y_correlations = np.corrcoef(X_np.T, y_np)[:-1, -1]
    
    print("\nCorrelations with Target:")
    for idx in feature_types['informative']:
        print(f"Informative feature {idx}: {y_correlations[idx]:.3f}")
        spearman_corr = stats.spearmanr(X_np[:, idx], y_np)[0]
        print(f"  - Spearman correlation: {spearman_corr:.3f}")
    
    print("\nMulticollinear Relationships:")
    for inf_idx in feature_types['informative']:
        print(f"\nCorrelations with informative feature {inf_idx}:")
        related_indices = [i for i in feature_types['multicollinear'] 
                         if abs(correlations[inf_idx, i]) > 0.5]
        for rel_idx in related_indices:
            print(f"Feature {rel_idx}: {correlations[inf_idx, rel_idx]:.3f}")