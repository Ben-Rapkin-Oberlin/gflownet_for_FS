from typing import Tuple
import torch
from gfn.preprocessors import Preprocessor
from gfn.states import States

class FeatureSelectionPreprocessor(Preprocessor):
    """Preprocessor for feature selection states."""
    
    def __init__(self, n_features: int):
        self.n_features = n_features
        self._output_shape = (n_features,)  # Each state is already a binary vector
        # Pass output dimension to parent class
        super().__init__(output_dim=n_features)
        
    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._output_shape
        
    def preprocess(self, states: States) -> torch.Tensor:
        """
        Preprocess states tensor.
        For feature selection, we just need to ensure it's a float tensor.
        
        Args:
            states: States object containing the states tensor
            
        Returns:
            torch.Tensor: Preprocessed tensor
        """
        if isinstance(states, States):
            states_tensor = states.tensor
        else:
            states_tensor = states
            
        return states_tensor.float()