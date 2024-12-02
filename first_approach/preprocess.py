from typing import Tuple
import torch
from gfn.preprocessors import Preprocessor

class FeatureSelectionPreprocessor(Preprocessor):
    """Preprocessor for feature selection states."""
    
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self._output_shape = (n_features,)  # Each state is already a binary vector
        
    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._output_shape
        
    def preprocess(self, states_tensor: torch.Tensor) -> torch.Tensor:
        """
        Preprocess states tensor.
        For feature selection, we just need to ensure it's a float tensor.
        """
        return states_tensor.float()