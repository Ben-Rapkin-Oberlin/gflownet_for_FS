import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from env import FeatureSelectionEnv
from gflownet import FeatureSelectionGFlowNet
class ModelEvaluator:
    """Evaluates feature sets using a random forest classifier."""
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def evaluate_features(self, feature_mask):
        """
        Evaluate a feature set using random forest classifier.
        
        Args:
            feature_mask: Binary array indicating selected features
        
        Returns:
            float: Classification accuracy
        """
        # Get selected feature indices
        selected_features = np.where(feature_mask == 1)[0]
        
        if len(selected_features) == 0:
            return 0.0
        
        # Train random forest on selected features
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train[:, selected_features], self.y_train)
        
        # Evaluate on test set
        score = rf.score(self.X_test[:, selected_features], self.y_test)
        return score

class FeatureSelectionEnvWithReward(FeatureSelectionEnv):
    """Extends FeatureSelectionEnv to include reward calculation."""
    
    def __init__(self, n_features, target_features, evaluator):
        super().__init__(n_features, target_features)
        self.evaluator = evaluator
    
    def log_reward(self, final_states):
        """Calculate log reward based on model performance."""
        rewards = []
        for state in final_states:
            # Convert to numpy for sklearn
            feature_mask = state.cpu().numpy()
            
            # Skip invalid states
            if not self.is_terminal_state(state.unsqueeze(0)):
                rewards.append(float('-inf'))
                continue
            
            # Evaluate feature set
            score = self.evaluator.evaluate_features(feature_mask)
            
            # Convert to log reward
            log_reward = np.log(max(score, 1e-10))
            rewards.append(log_reward)
        
        return torch.tensor(rewards, device=final_states.device)

def create_synthetic_dataset(n_samples=1000, n_features=20):
    """Create a synthetic dataset for testing."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,  # Only 10 features are actually informative
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def main():
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    X_train, X_test, y_train, y_test = create_synthetic_dataset(
        n_samples=1000,
        n_features=20
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(X_train, X_test, y_train, y_test)
    
    # Create environment with reward
    env = FeatureSelectionEnvWithReward(
        n_features=20,      # Total features
        target_features=10,  # We want to select 10 features
        evaluator=evaluator
    )
    
    # Initialize GFlowNet
    print("Initializing GFlowNet...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gfn = FeatureSelectionGFlowNet(
        n_features=20,
        target_features=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        lambda_=0.9
    ).to(device)
    
    # Create optimizer
    optimizer = gfn.get_optimizer(lr_transformer=1e-4, lr_heads=1e-3)
    
    # Training loop
    n_epochs = 100
    batch_size = 16
    best_score = float('-inf')
    best_features = None
    
    print("Starting training...")
    for epoch in tqdm(range(n_epochs)):
        # Sample trajectories
        trajectories = gfn.sampler.sample_trajectories(env, n=batch_size)
        
        # Calculate loss
        loss = gfn.gflownet.loss(env, trajectories)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Every 10 epochs, evaluate current best feature set
        if epoch % 10 == 0:
            with torch.no_grad():
                # Sample multiple feature sets
                feature_sets = gfn.sample_feature_sets(n_samples=5)
                
                # Evaluate each feature set
                for feature_set in feature_sets:
                    score = evaluator.evaluate_features(feature_set.cpu().numpy())
                    
                    if score > best_score:
                        best_score = score
                        best_features = feature_set.cpu().numpy()
                
                print(f"\nEpoch {epoch}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Best score so far: {best_score:.4f}")
                print("Best features:", np.where(best_features == 1)[0].tolist())
    
    print("\nTraining completed!")
    print("Final best score:", best_score)
    print("Selected features:", np.where(best_features == 1)[0].tolist())
    
    # Evaluate all features baseline
    all_features_score = evaluator.evaluate_features(np.ones(20))
    print("Score with all features:", all_features_score)

if __name__ == "__main__":
    main()