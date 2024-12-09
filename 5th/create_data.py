import numpy as np
from scipy import stats

def create_synthetic_dataset(
    n_samples=1000,
    n_features=50,
    n_informative=2,
    n_multicollinear=3,
    correlation_strength=0.7,
    noise_level=0.1,           # Base noise level for target
    multicollinear_noise=0.1,  # Noise in multicollinear relationships
    feature_noise=0.1,         # Base noise level for features
    noise_config={
        'global_noise_scale': 1.0,    # Scales all noise in the dataset
        'feature_noise_scale': 1.0,   # Additional scaling for feature noise
        'target_noise_scale': 1.0,    # Additional scaling for target noise
        'informative_noise': 0.1,     # Noise added to informative features
        'noise_feature_std': 1.0,     # Standard deviation of noise features
        'signal_to_noise_ratio': None # If set, adjusts noise to match desired SNR
    },
    nonlinear_features=None,
    interaction_features=None,
    missing_data=None,
    feature_distributions=None,
    outlier_config=None,
    heteroscedastic_noise=None,
    random_state=42
):
    """
    Enhanced synthetic dataset generator with comprehensive noise control.
    
    New Noise Parameters:
    --------------------
    feature_noise : float
        Base noise level for all features
    
    noise_config : dict
        global_noise_scale : float
            Master scaling factor for all noise in the dataset
        feature_noise_scale : float
            Additional scaling factor specifically for feature noise
        target_noise_scale : float
            Additional scaling factor specifically for target noise
        informative_noise : float
            Noise level specifically for informative features
        noise_feature_std : float
            Standard deviation of pure noise features
        signal_to_noise_ratio : float or None
            If set, adjusts noise to achieve desired SNR
    """
    rng = np.random.RandomState(random_state)
    
    # Apply global noise scaling
    effective_noise_level = noise_level * noise_config['global_noise_scale'] * noise_config['target_noise_scale']
    effective_feature_noise = feature_noise * noise_config['global_noise_scale'] * noise_config['feature_noise_scale']
    effective_multicollinear_noise = multicollinear_noise * noise_config['global_noise_scale'] * noise_config['feature_noise_scale']
    
    # Initialize feature matrix with controlled random noise
    X = rng.normal(0, noise_config['noise_feature_std'], (n_samples, n_features))
    
    # Create informative features with specified distributions and noise
    informative_indices = list(range(n_informative))
    informative_features = np.zeros((n_samples, n_informative))
    
    if feature_distributions:
        for idx in range(n_informative):
            dist = feature_distributions.get(idx, 'normal')
            base_feature = None
            
            if dist == 'normal':
                base_feature = rng.randn(n_samples)
            elif dist == 'student_t':
                base_feature = stats.t.rvs(df=3, size=n_samples, random_state=rng)
            elif dist == 'lognormal':
                base_feature = rng.lognormal(0, 1, n_samples)
            elif dist == 'categorical':
                base_feature = rng.choice(3, size=n_samples)
                
            # Add controlled noise to informative features
            noise = rng.normal(0, noise_config['informative_noise'], n_samples)
            informative_features[:, idx] = base_feature + noise
    else:
        base_features = rng.randn(n_samples, n_informative)
        noise = rng.normal(0, noise_config['informative_noise'], (n_samples, n_informative))
        informative_features = base_features + noise

    # Create base target variable
    y = np.zeros(n_samples)
    for i in range(n_informative):
        y += correlation_strength * informative_features[:, i]
    
    # Add non-linear relationships
    if nonlinear_features:
        for feature_type, indices in nonlinear_features.items():
            for idx in indices:
                if idx < n_informative:
                    if feature_type == 'polynomial':
                        y += 0.3 * informative_features[:, idx]**2
                    elif feature_type == 'exponential':
                        y += 0.3 * np.exp(informative_features[:, idx] / 2)
                    elif feature_type == 'periodic':
                        y += 0.3 * np.sin(informative_features[:, idx] * 2 * np.pi)

    # Add interaction effects
    if interaction_features:
        for i, j in interaction_features:
            if i < n_informative and j < n_informative:
                y += 0.2 * informative_features[:, i] * informative_features[:, j]

    # Create multicollinear features with controlled noise
    multicollinear_indices = []
    current_idx = n_informative
    
    for inf_idx in range(n_informative):
        related_indices = list(range(current_idx, current_idx + n_multicollinear))
        multicollinear_indices.extend(related_indices)
        
        for j in range(n_multicollinear):
            correlation = rng.uniform(0.7, 0.9)
            noise = rng.normal(0, effective_multicollinear_noise, n_samples)
            X[:, related_indices[j]] = (correlation * informative_features[:, inf_idx] + 
                                      np.sqrt(1 - correlation**2) * noise)
        
        current_idx += n_multicollinear
    
    # Place informative features in the matrix
    X[:, informative_indices] = informative_features
    
    # Add noise to target based on configuration
    if noise_config['signal_to_noise_ratio'] is not None:
        # Calculate signal power
        signal_power = np.var(y)
        # Calculate required noise power for desired SNR
        desired_noise_power = signal_power / noise_config['signal_to_noise_ratio']
        # Adjust noise level to achieve desired SNR
        effective_noise_level = np.sqrt(desired_noise_power)
    
    # Add heteroscedastic noise if configured
    if heteroscedastic_noise:
        dep_feature = heteroscedastic_noise['dependent_feature']
        noise_factor = heteroscedastic_noise['noise_factor'] * noise_config['global_noise_scale']
        variance = np.exp(X[:, dep_feature])
        noise = rng.normal(0, noise_factor * variance)
        y += noise
    else:
        # Add homoscedastic noise
        y += rng.normal(0, effective_noise_level, n_samples)
    
    # Add outliers if configured
    if outlier_config:
        n_outliers = int(outlier_config['contamination'] * n_samples)
        outlier_idx = rng.choice(n_samples, n_outliers, replace=False)
        X[outlier_idx] *= outlier_config['magnitude']
        y[outlier_idx] *= outlier_config['magnitude']
    
    # Introduce missing values if configured
    if missing_data:
        mask = rng.random(X.shape) < missing_data['rate']
        if missing_data['mechanism'] == 'MAR':
            mask = mask & (X[:, 0] > X[:, 0].mean())
        X[mask] = np.nan
    
    # Track feature types
    feature_types = {
        'informative': informative_indices,
        'multicollinear': multicollinear_indices,
        'noise': list(set(range(n_features)) - 
                     set(informative_indices) - 
                     set(multicollinear_indices))
    }
    
    # Standardize all features (excluding NaN values if present)
    X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    y = (y - y.mean()) / y.std()
    
    return X, y, feature_types