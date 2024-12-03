import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import random
from gflownet import ImprovedGFlowNet


import cProfile
import pstats

def set_all_seeds(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def create_synthetic_dataset(n_samples=1000, n_features=20, n_informative=5, random_state=42):
    """Create a synthetic dataset with informative and noise features"""
    rng = np.random.RandomState(random_state)
    
    informative_features = rng.randn(n_samples, n_informative)
    noise_features = rng.randn(n_samples, n_features - n_informative)
    
    X = np.hstack([informative_features, noise_features])
    y = (informative_features[:, 0] + informative_features[:, 1] > 0).astype(int)
    
    return X, y

def evaluate_feature_subset(X, y, feature_indices, scaler=None, clf=None):
    """Fast evaluation using pre-initialized objects"""
    X_selected = X[:, feature_indices]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected.cpu(), y.cpu(), test_size=0.2, random_state=42
    )

    if scaler is None:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train.numpy())
    X_test = scaler.transform(X_test.numpy())

    if clf is None:
        clf = LogisticRegression(max_iter=100, solver='lbfgs')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)
def main():
    SEED = 42
    set_all_seeds(SEED)
        
    # Force CUDA device if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
            torch.cuda.init()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
            print("Using CPU")
        
    # Create synthetic dataset and model
    n_samples = 1000
    n_features = 20
    n_select = 5
    n_episodes = 21
    batch_size = 64  # Increased batch size for GPU
        
    X, y = create_synthetic_dataset(n_samples=n_samples, n_features=n_features)
        
    # Move data preprocessing to GPU if available
    if device.type == 'cuda':
          scaler = StandardScaler().fit(X)
          X = torch.tensor(scaler.transform(X), device=device, dtype=torch.float32)
          y = torch.tensor(y, device=device)
    else:
          scaler = StandardScaler()
            
    clf = LogisticRegression(max_iter=100, solver='lbfgs', random_state=SEED)
        
    model = ImprovedGFlowNet(
            num_elements=n_features,
            target_size=n_select,
            hidden_dim=128,
            num_heads=4,
            num_layers=3,
            device=device
        )
    
    # Training loop with improvements
    print(f"Training GFlowNet on {device}...")
    
    start_time = time.time()
    best_reward = 0
    
    for episode in range(n_episodes):
        temp = max(1.0 - episode/n_episodes, 0.1)
        
        trajectories = []
        rewards = []
        
        # Use beam search occasionally
        use_beam_search = episode % 10 == 0
        
        for _ in range(batch_size):
            subset, _ = model.sample_subset(temperature=temp, use_beam_search=use_beam_search)
            accuracy = evaluate_feature_subset(X, y, subset, scaler, clf)
            trajectories.append(subset)
            rewards.append(torch.tensor(accuracy, device=device))
            model.update_feature_stats(subset, accuracy)
            
            if accuracy > best_reward:
                best_reward = accuracy
                print(f"New best reward: {best_reward:.4f}")
        
        loss = model.train_step(trajectories, rewards, temp)
        
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}, Loss: {loss:.4f}, "
                  f"Best Reward: {best_reward:.4f}, "
                  f"Temp: {temp:.4f}, Time: {elapsed:.1f}s")
    
    model.load_best_model()
    
    # Final evaluation
    print("\nFinal evaluation using best model:")
    final_subsets = []
    final_accuracies = []
    
    for _ in range(10):
        subset, _ = model.sample_subset(temperature=0.1, use_beam_search=True)
        accuracy = evaluate_feature_subset(X, y, subset, scaler, clf)
        final_subsets.append(subset)
        final_accuracies.append(accuracy)
        print(f"Selected features: {subset}, Accuracy: {accuracy:.4f}")
    
    # Print final statistics
    avg_accuracy = np.mean(final_accuracies)
    std_accuracy = np.std(final_accuracies)
    print(f"\nAverage accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    
    # Feature importance analysis
    feature_importance = model.get_feature_importance()
    top_features = np.argsort(feature_importance)[::-1][:n_select]
    print("\nTop features by importance:")
    for idx, feature in enumerate(top_features):
        print(f"Feature {feature}: {feature_importance[feature]:.4f}")

if __name__ == "__main__":
    cProfile.run('main()', 'profile_stats')
    p = pstats.Stats('profile_stats')
    p.sort_stats('tottime').print_stats(10)