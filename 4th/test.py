# feature_selection_comparison_real_data.py

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """
    Load and clean the communities_and_crime dataset.
    Returns:
        X (pd.DataFrame): Cleaned feature set.
        y (pd.Series): Target variable.
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
    num_imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    # Impute categorical columns with most frequent
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # Convert data types
    X[numerical_cols] = X[numerical_cols].astype(float)
    X[categorical_cols] = X[categorical_cols].astype(str)
    
    # Encode categorical variables using OneHotEncoder
    if categorical_cols:
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_cats = encoder.fit_transform(X[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cols, index=X.index)
        X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Final checks
    assert X.isnull().sum().sum() == 0, "There are still missing values."
    
    return X, y

def train_and_evaluate(X, y, selected_features):
    """
    Train a Linear Regression model on selected features and evaluate MSE.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        selected_features (list): List of selected feature names.
    Returns:
        mse (float): Mean Squared Error on the test set.
    """
    X_selected = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def select_kbest(X, y, k=10, score_func=f_regression):
    """
    Select K best features using univariate statistical tests.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        k (int): Number of top features to select.
        score_func: Scoring function.
    Returns:
        selected_features (list): List of selected feature names.
        mse (float): Mean Squared Error of the model trained on selected features.
    """
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    mse = train_and_evaluate(X, y, selected_features)
    return selected_features, mse

def recursive_feature_elimination(X, y, k=10):
    """
    Select features using Recursive Feature Elimination (RFE).
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        k (int): Number of features to select.
    Returns:
        selected_features (list): List of selected feature names.
        mse (float): Mean Squared Error of the model trained on selected features.
    """
    model = LinearRegression()
    selector = RFE(model, n_features_to_select=k, step=1)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_].tolist()
    mse = train_and_evaluate(X, y, selected_features)
    return selected_features, mse

def feature_importance_selection(X, y, k=10):
    """
    Select features based on feature importance from a Random Forest Regressor.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        k (int): Number of top features to select.
    Returns:
        selected_features (list): List of selected feature names.
        mse (float): Mean Squared Error of the model trained on selected features.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = X.columns[indices[:k]].tolist()
    mse = train_and_evaluate(X, y, selected_features)
    return selected_features, mse

def sequential_forward_selection(X, y, k=10):
    """
    Select features using Sequential Forward Selection.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        k (int): Number of features to select.
    Returns:
        selected_features (list): List of selected feature names.
        mse (float): Mean Squared Error of the model trained on selected features.
    """
    model = LinearRegression()
    selector = SequentialFeatureSelector(
        model, n_features_to_select=k, direction='forward', scoring='neg_mean_squared_error', n_jobs=-1
    )
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    mse = train_and_evaluate(X, y, selected_features)
    return selected_features, mse

def correlation_based_selection(X, y, k=10, threshold=0.05):
    """
    Select features based on correlation with the target variable.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        k (int): Number of top features to select.
        threshold (float): Minimum correlation required.
    Returns:
        selected_features (list): List of selected feature names.
        mse (float): Mean Squared Error of the model trained on selected features.
    """
    correlation = X.corrwith(y).abs()
    selected_features = correlation.sort_values(ascending=False).head(k).index.tolist()
    mse = train_and_evaluate(X, y, selected_features)
    return selected_features, mse

def main():
    # Load and clean data
    X, y = load_and_clean_data()
    print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    
    # Define number of features to select
    k = 10
    
    # Dictionary to store results
    results = {}
    
    # 1. Univariate Feature Selection with f_regression
    features, mse = select_kbest(X, y, k=k, score_func=f_regression)
    results['SelectKBest (f_regression)'] = {'features': features, 'mse': mse}
    print(f"SelectKBest (f_regression) - MSE: {mse:.4f}")
    
    # 2. Univariate Feature Selection with mutual_info_regression
    features, mse = select_kbest(X, y, k=k, score_func=mutual_info_regression)
    results['SelectKBest (mutual_info_regression)'] = {'features': features, 'mse': mse}
    print(f"SelectKBest (mutual_info_regression) - MSE: {mse:.4f}")
    
    # 3. Recursive Feature Elimination (RFE)
    #features, mse = recursive_feature_elimination(X, y, k=k)
    #results['RFE'] = {'features': features, 'mse': mse}
    #print(f"RFE - MSE: {mse:.4f}")
    
    # 4. Feature Importance from Random Forest
    features, mse = feature_importance_selection(X, y, k=k)
    results['Random Forest Importance'] = {'features': features, 'mse': mse}
    print(f"Random Forest Importance - MSE: {mse:.4f}")
    
    # 5. Sequential Forward Selection
    #features, mse = sequential_forward_selection(X, y, k=k)
    #results['Sequential Forward Selection'] = {'features': features, 'mse': mse}
    #print(f"Sequential Forward Selection - MSE: {mse:.4f}")
    
    # 6. Correlation-Based Selection
    features, mse = correlation_based_selection(X, y, k=k)
    results['Correlation-Based Selection'] = {'features': features, 'mse': mse}
    print(f"Correlation-Based Selection - MSE: {mse:.4f}")
    
    # Display comparison
    print("\n=== Feature Selection Comparison ===")
    for method, data in results.items():
        print(f"{method}: MSE = {data['mse']:.4f}")
    
    # Optionally, visualize the results
    methods = list(results.keys())
    mses = [data['mse'] for data in results.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, mses, color='skyblue')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Feature Selection Methods Comparison (Top {k} Features)')
    plt.xticks(rotation=45, ha='right')
    
    # Annotate bars with MSE values
    for bar, mse in zip(bars, mses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{mse:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    main()
