import torch
import numpy as np
from typing import Tuple, Optional

class FeatureSelectionEnv:
    """
    A discrete environment for feature selection where:
    - States are binary vectors representing selected features
    - Actions are integers representing which feature to select next
    - Terminal states are those with exactly m features selected
    """
    def __init__(self, n_features: int, target_features: int):
        """
        Args:
            n_features: Total number of available features
            target_features: Number of features to select (m)
        """
        self.n_features = n_features
        self.target_features = target_features
        
        # Define initial state (no features selected)
        self.s0 = torch.zeros(n_features, dtype=torch.float32)
        
        # Define sink state (used for padding)
        self.sf = torch.full((n_features,), -1, dtype=torch.float32)
        
        # Number of actions = n_features + 1 (one per feature + exit action)
        self.n_actions = n_features + 1
        self.action_shape = (1,)

    def step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward step in the environment."""
        next_states = states.clone()
        
        # Handle non-exit actions (feature selection)
        non_exit_mask = actions[:, 0] != self.n_actions - 1
        if non_exit_mask.any():
            feature_indices = actions[non_exit_mask, 0]
            next_states[non_exit_mask, feature_indices] = 1
            
        # Handle exit actions
        exit_mask = ~non_exit_mask
        if exit_mask.any():
            next_states[exit_mask] = self.sf
            
        return next_states
    
    def backward_step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Backward step in the environment."""
        prev_states = states.clone()
        
        # Handle non-exit actions (feature deselection)
        non_exit_mask = actions[:, 0] != self.n_actions - 1
        if non_exit_mask.any():
            feature_indices = actions[non_exit_mask, 0]
            prev_states[non_exit_mask, feature_indices] = 0
            
        return prev_states

    def update_masks(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update forward and backward masks based on current states."""
        batch_size = states.shape[0]
        
        # Initialize masks
        forward_masks = torch.ones(batch_size, self.n_actions, dtype=torch.bool)
        backward_masks = torch.ones(batch_size, self.n_actions - 1, dtype=torch.bool)
        
        # Count selected features in each state
        n_selected = torch.sum(states == 1, dim=1)
        
        # Forward masks: can't select already selected features
        forward_masks[:, :-1] = states != 1
        
        # Can only exit when exactly target_features are selected
        forward_masks[:, -1] = (n_selected == self.target_features)
        
        # Can't take any action if we're at sink state
        is_sink = torch.all(states == self.sf, dim=1)
        forward_masks[is_sink] = False
        
        # Backward masks: can only deselect selected features
        backward_masks = states == 1
        
        return forward_masks, backward_masks

    def is_terminal_state(self, states: torch.Tensor) -> torch.Tensor:
        """Check if states are terminal (have exactly target_features selected)."""
        n_selected = torch.sum(states == 1, dim=1)
        return n_selected == self.target_features

    def make_random_states_tensor(self, batch_shape: Tuple) -> torch.Tensor:
        """Create random valid states for initialization."""
        states = torch.zeros((*batch_shape, self.n_features))
        
        # Randomly select target_features number of features for each state
        for i in range(batch_shape[0]):
            indices = np.random.choice(self.n_features, 
                                     size=self.target_features, 
                                     replace=False)
            states[i, indices] = 1
            
        return states

    def log_reward(self, final_states: torch.Tensor) -> torch.Tensor:
        """
        Calculate log reward for terminal states.
        This should be overridden with actual model performance evaluation.
        """
        # Placeholder - return uniform reward for all valid terminal states
        is_valid = self.is_terminal_state(final_states)
        return torch.where(is_valid, 
                         torch.tensor(0.0),
                         torch.tensor(float('-inf')))