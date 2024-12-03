import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from gfn.states import DiscreteStates
from gfn.env import DiscreteEnv
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.preprocessors import Preprocessor
from gfn.gflownet import SubTBGFlowNet
from gfn.utils.modules import MLP
from gfn.utils.common import set_seed

class FeatureSelectionStates(DiscreteStates):
    """States class for feature selection"""
    def __init__(self, tensor, env):
        super().__init__(tensor=tensor)
        self.env = env
        
    def update_masks(self):
        """Update forward and backward masks"""
        self.forward_masks, self.backward_masks = self.env.update_masks(self.tensor)

class FeatureSelectionEnv(DiscreteEnv):
    """Simple discrete environment for feature selection"""
    def __init__(self, n_features: int, target_features: int, device_str: str = "cpu"):
        self.n_features = n_features
        self.target_features = target_features
        self.device = torch.device(device_str)
        
        # Initial state - no features selected
        s0 = torch.zeros(n_features, dtype=torch.float32, device=self.device)
        
        # Sink state
        sf = torch.full((n_features,), -1, dtype=torch.float32, device=self.device)
        
        # Number of actions = number of features + exit action
        n_actions = n_features + 1
        
        # Define state shape
        state_shape = (n_features,)
        
        super().__init__(
            n_actions=n_actions,
            s0=s0,
            state_shape=state_shape,
            sf=sf,
            device_str=device_str
        )
        
        # Initialize preprocessor
        self.preprocessor = FeatureSelectionPreprocessor(n_features)
        
    def make_states_class(self):
        """Create states class with proper initialization"""
        env = self
        
        class States(DiscreteStates):
            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            n_actions = env.n_actions
            device = env.device

            def update_masks(self):
                self.forward_masks, self.backward_masks = env.update_masks(self.tensor)
                
        return States

    def update_masks(self, states) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update forward and backward masks"""
        # Handle both States objects and raw tensors
        if hasattr(states, 'tensor'):
            states_tensor = states.tensor
        else:
            states_tensor = states
            
        batch_size = states_tensor.shape[0]
        
        # Initialize masks
        forward_masks = torch.zeros(batch_size, self.n_actions, dtype=torch.bool, device=states.device)
        backward_masks = torch.zeros(batch_size, self.n_actions - 1, dtype=torch.bool, device=states.device)
        
        # Get state info
        n_selected = torch.sum(states_tensor == 1, dim=1)
        is_sink = torch.all(states_tensor == self.sf, dim=1)
        
        # Set forward masks
        can_select = ~is_sink & (n_selected < self.target_features)
        if can_select.any():
            forward_masks[can_select, :-1] = states_tensor[can_select] != 1  # Can select unselected features
        
        # Can exit at target
        can_exit = ~is_sink & (n_selected == self.target_features)
        if can_exit.any():
            forward_masks[can_exit, -1] = True
        
        # Set backward masks
        non_sink = ~is_sink
        if non_sink.any():
            backward_masks[non_sink] = states_tensor[non_sink] == 1  # Can only deselect selected features
        
        return forward_masks, backward_masks

    def step(self, states, actions) -> torch.Tensor:
        """Forward step in environment"""
        if hasattr(states, 'tensor'):
            states_tensor = states.tensor
        else:
            states_tensor = states
            
        next_states = states_tensor.clone()
        
        # Don't modify sink states
        is_sink = torch.all(states_tensor == self.sf, dim=1)
        if is_sink.all():
            return next_states
        
        # Get actions tensor
        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor
        else:
            actions_tensor = actions
            
        if actions_tensor.dim() == 1:
            actions_tensor = actions_tensor.unsqueeze(-1)
            
        # For non-sink states
        n_selected = torch.sum(next_states == 1, dim=1)
        
        # Handle exit action
        exit_mask = actions_tensor[..., 0] == self.n_actions - 1
        at_target = n_selected == self.target_features
        if (exit_mask & at_target).any():
            next_states[exit_mask & at_target] = self.sf
        
        # Handle feature selection
        can_select = ~is_sink & (n_selected < self.target_features) & ~exit_mask
        if can_select.any():
            valid_actions = torch.clamp(actions_tensor[can_select, 0], 0, self.n_features - 1).long()
            batch_indices = torch.arange(len(next_states), device=next_states.device)[can_select]
            next_states[batch_indices, valid_actions] = 1
            
        return next_states

    def backward_step(self, states, actions) -> torch.Tensor:
        """Backward step in environment - removing selected features"""
        if hasattr(states, 'tensor'):
            states_tensor = states.tensor
        else:
            states_tensor = states
            
        prev_states = states_tensor.clone()
        
        # Don't modify sink states
        is_sink = torch.all(states_tensor == self.sf, dim=1)
        if is_sink.all():
            return prev_states
            
        # Get actions tensor - ensure it's a valid feature index
        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor
        else:
            actions_tensor = actions
            
        if actions_tensor.dim() == 1:
            actions_tensor = actions_tensor.unsqueeze(-1)
            
        # Handle feature deselection for non-sink states
        non_sink = ~is_sink
        if non_sink.any():
            # Ensure actions are valid feature indices
            feature_indices = torch.clamp(actions_tensor[non_sink, 0], 0, self.n_features - 1).long()
            batch_indices = torch.arange(len(prev_states), device=prev_states.device)[non_sink]
            # Only deselect if feature was selected
            valid_deselection = prev_states[batch_indices, feature_indices] == 1
            prev_states[batch_indices[valid_deselection], feature_indices[valid_deselection]] = 0
            
        return prev_states

    def log_reward(self, states) -> torch.Tensor:
        """Calculate log reward for terminal states"""
        if hasattr(states, 'tensor'):
            states_tensor = states.tensor
        else:
            states_tensor = states
            
        n_selected = torch.sum(states_tensor == 1, dim=1)
        is_valid = n_selected == self.target_features
        
        return torch.where(is_valid, 
                         torch.tensor(0.0, device=states_tensor.device),
                         torch.tensor(float('-inf'), device=states_tensor.device))

class FeatureSelectionPreprocessor(Preprocessor):
    """Simple preprocessor for feature selection states"""
    def __init__(self, n_features: int):
        super().__init__(output_dim=n_features)
        self.n_features = n_features
        
    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self.n_features,)
        
    def preprocess(self, states) -> torch.Tensor:
        """Convert states to float tensor"""
        if hasattr(states, 'tensor'):
            states = states.tensor
        return states.float()

class ModelEvaluator:
    """Evaluates feature sets using random forest"""
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def evaluate_features(self, feature_mask):
        selected = np.where(feature_mask == 1)[0]
        if len(selected) == 0:
            return 0.0
            
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train[:, selected], self.y_train)
        return rf.score(self.X_test[:, selected], self.y_test)

def create_synthetic_dataset(n_samples=1000, n_features=20):
    """Create synthetic dataset for testing"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        random_state=42
    )
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating synthetic dataset...")
    X_train, X_test, y_train, y_test = create_synthetic_dataset()
    
    # Create environment and evaluator
    print("Setting up environment...")
    evaluator = ModelEvaluator(X_train, X_test, y_train, y_test)
    env = FeatureSelectionEnv(
        n_features=20,
        target_features=10,
        device_str=str(device)
    )
    
    # Create policy networks
    print("Creating GFlowNet...")
    hidden_dim = 128
    n_hidden = 2
    
    # Forward policy can select any feature or exit (n_features + 1 actions)
    pf_module = MLP(
        input_dim=env.preprocessor.output_dim,  # n_features
        output_dim=env.n_actions,              # n_features + 1
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden
    )
    
    # Backward policy can only deselect features (n_features actions)
    pb_module = MLP(
        input_dim=env.preprocessor.output_dim,  # n_features
        output_dim=env.n_features,             # n_features (no exit action)
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
        trunk=pf_module.trunk  # Share trunk network
    )
    
    # Log flow network
    logF_module = MLP(
        input_dim=env.preprocessor.output_dim,  # n_features
        output_dim=1,                          # scalar output
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden,
        trunk=pf_module.trunk  # Share trunk network
    )
    
    # Create estimators with explicitly defined action spaces
    pf = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,  # n_features + 1 (select feature or exit)
        preprocessor=env.preprocessor,
        is_backward=False
    )
    
    pb = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions - 1,  # n_features (deselect feature)
        preprocessor=env.preprocessor,
        is_backward=True
    )
    
    logF = ScalarEstimator(
        module=logF_module,
        preprocessor=env.preprocessor
    )
    
    # Create GFlowNet
    gflownet = SubTBGFlowNet(
        pf=pf, 
        pb=pb,
        logF=logF,
        lamda=0.9
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam([
        {'params': gflownet.pf_pb_parameters(), 'lr': 1e-4},
        {'params': gflownet.logF_parameters(), 'lr': 1e-3}
    ])
    
    # Training loop
    n_epochs = 1000
    batch_size = 32
    best_score = float('-inf')
    best_features = None
    
    print("Starting training...")
    for epoch in tqdm(range(n_epochs)):
        # Sample trajectories
        trajectories = gflownet.sample_trajectories(env, n=batch_size)
        training_samples = gflownet.to_training_samples(trajectories)
        
        # Calculate loss and update
        optimizer.zero_grad()
        loss = gflownet.loss(env, training_samples)
        loss.backward()
        optimizer.step()
        
        # Evaluate periodically
        if epoch % 50 == 0:
            with torch.no_grad():
                terminal_states = trajectories.states.tensor[trajectories.done]
                
                for state in terminal_states:
                    score = evaluator.evaluate_features(state.cpu().numpy())
                    if score > best_score:
                        best_score = score
                        best_features = state.cpu().numpy()
                
                print(f"\nEpoch {epoch}")
                print(f"Loss: {loss.item():.4f}")
                print(f"Best score: {best_score:.4f}")
                if best_features is not None:
                    print(f"Best features: {np.where(best_features == 1)[0].tolist()}")
    
    print("\nTraining completed!")
    print(f"Final best score: {best_score:.4f}")
    if best_features is not None:
        print(f"Selected features: {np.where(best_features == 1)[0].tolist()}")
    
    # Baseline comparison
    all_features_score = evaluator.evaluate_features(np.ones(20))
    print(f"Score with all features: {all_features_score:.4f}")

if __name__ == "__main__":
    main()