import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FeatureSelectionState:
    """Represents the state in feature selection process."""
    selected_features: List[int]  # Indices of selected features
    available_features: List[int]  # Indices of features still available
    
    def __eq__(self, other):
        if not isinstance(other, FeatureSelectionState):
            return False
        return (self.selected_features == other.selected_features and 
                self.available_features == other.available_features)

class FeatureSelectionEnv:
    """Environment for feature selection using GFlowNet."""
    
    def __init__(
        self, 
        n_features: int,
        target_size: int,
        reward_fn,
        device: str = "cpu"
    ):
        """
        Args:
            n_features: Total number of features to select from
            target_size: Number of features to select
            reward_fn: Function that takes list of selected features and returns reward
            device: Device to use for computations
        """
        self.n_features = n_features
        self.target_size = target_size
        self.reward_fn = reward_fn
        self.device = device
        
        # Define initial state
        self.initial_state = FeatureSelectionState(
            selected_features=[],
            available_features=list(range(n_features))
        )
        
        # Number of possible actions is max number of available features
        self.n_actions = n_features
        
    def get_next_state(
        self, 
        state: FeatureSelectionState, 
        action: int
    ) -> Optional[FeatureSelectionState]:
        """Get next state after taking an action."""
        if action not in state.available_features:
            return None
            
        next_selected = state.selected_features + [action]
        next_available = [f for f in state.available_features if f != action]
        
        return FeatureSelectionState(
            selected_features=next_selected,
            available_features=next_available
        )
    
    def get_valid_actions(self, state: FeatureSelectionState) -> List[int]:
        """Get list of valid actions from current state."""
        return state.available_features
    
    def is_terminal(self, state: FeatureSelectionState) -> bool:
        """Check if state is terminal."""
        return len(state.selected_features) == self.target_size
    
    def get_reward(self, state: FeatureSelectionState) -> float:
        """Get reward for terminal state."""
        if not self.is_terminal(state):
            return 0.0
        return self.reward_fn(state.selected_features)
    
    def preprocess_state(self, state: FeatureSelectionState) -> torch.Tensor:
        """Convert state to tensor representation."""
        # Create binary vector where 1s indicate selected features
        state_tensor = torch.zeros(self.n_features, device=self.device)
        for idx in state.selected_features:
            state_tensor[idx] = 1.0
        return state_tensor
        
    def get_backward_action(
        self, 
        state: FeatureSelectionState
    ) -> Optional[Tuple[FeatureSelectionState, int]]:
        """Get previous state and action that led to current state."""
        if not state.selected_features:
            return None
            
        prev_selected = state.selected_features[:-1]
        prev_available = state.available_features + [state.selected_features[-1]]
        
        prev_state = FeatureSelectionState(
            selected_features=prev_selected,
            available_features=prev_available
        )
        
        action = state.selected_features[-1]
        return prev_state, action

    @property
    def preprocessor(self):
        """Return identity preprocessor for compatibility."""
        return lambda x: x