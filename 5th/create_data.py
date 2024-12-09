import numpy as np
from scipy import stats

def create_synthetic_dataset(
    n_samples=1000,
    n_features=50,
    n_informative=2,
    n_multicollinear=3,
    correlation_strength=0.7,
    noise_level=0.1,
    multicollinear_noise=0.1,
    nonlinear_features=None,
    interaction_features=None,
    missing_data=None,
    feature_distributions=None,
    outlier_config=None,
    heteroscedastic_noise=None,
    random_state=42
):
    """
    Enhanced synthetic dataset generator with additional complex relationships.
    
    New Parameters:
    --------------
    nonlinear_features : dict or None
        Config for non-linear relationships, e.g.,
        {'polynomial': [0], 'exponential': [1], 'periodic': [2]}
    interaction_features : list of tuples or None
        Features to create interactions between, e.g., [(0,1), (1,2)]
    missing_data : dict or None
        Config for missing values, e.g.,
        {'mechanism': 'MCAR', 'rate': 0.1}
    feature_distributions : dict or None
        Distribution type for features, e.g.,
        {0: 'normal', 1: 'student_t', 2: 'lognormal', 3: 'categorical'}
    outlier_config : dict or None
        Config for outliers, e.g.,
        {'contamination': 0.01, 'magnitude': 5}
    heteroscedastic_noise : dict or None
        Config for variance increases, e.g.,
        {'dependent_feature': 0, 'noise_factor': 0.1}
    """
    rng = np.random.RandomState(random_state)
    
    # Initialize feature matrix with random noise
    X = rng.randn(n_samples, n_features)
    
    # Create informative features with specified distributions
    informative_indices = list(range(n_informative))
    informative_features = np.zeros((n_samples, n_informative))
    
    if feature_distributions:
        for idx in range(n_informative):
            dist = feature_distributions.get(idx, 'normal')
            if dist == 'normal':
                informative_features[:, idx] = rng.randn(n_samples)
            elif dist == 'student_t':
                informative_features[:, idx] = stats.t.rvs(df=3, size=n_samples, random_state=rng)
            elif dist == 'lognormal':
                informative_features[:, idx] = rng.lognormal(0, 1, n_samples)
            elif dist == 'categorical':
                informative_features[:, idx] = rng.choice(3, size=n_samples)
    else:
        informative_features = rng.randn(n_samples, n_informative)

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

    # Create multicollinear features
    multicollinear_indices = []
    current_idx = n_informative
    
    for inf_idx in range(n_informative):
        related_indices = list(range(current_idx, current_idx + n_multicollinear))
        multicollinear_indices.extend(related_indices)
        
        for j in range(n_multicollinear):
            correlation = rng.uniform(0.7, 0.9)
            noise = rng.normal(0, multicollinear_noise, n_samples)
            X[:, related_indices[j]] = (correlation * informative_features[:, inf_idx] + 
                                      np.sqrt(1 - correlation**2) * noise)
        
        current_idx += n_multicollinear
    
    # Place informative features in the matrix
    X[:, informative_indices] = informative_features
    
    # Add heteroscedastic noise if configured
    if heteroscedastic_noise:
        dep_feature = heteroscedastic_noise['dependent_feature']
        noise_factor = heteroscedastic_noise['noise_factor']
        variance = np.exp(X[:, dep_feature])
        noise = rng.normal(0, noise_factor * variance)
        y += noise
    else:
        # Add homoscedastic noise
        y += rng.normal(0, noise_level, n_samples)
    
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
            # Make missing rate dependent on first informative feature
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