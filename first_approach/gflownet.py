import torch
import torch.nn as nn
from gfn.gflownet import SubTBGFlowNet
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.samplers import Sampler

from preprocess import FeatureSelectionPreprocessor
from env import FeatureSelectionEnv
from trans import FeatureSelectionTransformer, TransformerPolicyHead, TransformerLogFHead

class FeatureSelectionGFlowNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        target_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        lambda_: float = 0.9,
        temperature: float = .5
    ):
        super().__init__()
        self.temperature = max(temperature, 1e-3) 
        
        # Initialize environment
        self.env = FeatureSelectionEnv(n_features, target_features)
        
        # Add preprocessor to environment
        self.env.preprocessor = FeatureSelectionPreprocessor(n_features)
        
        # Create shared transformer backbone
        self.transformer = FeatureSelectionTransformer(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        # Create policy and logF heads
        self.pf_head = TransformerPolicyHead(d_model, self.env.n_actions)
        self.pb_head = TransformerPolicyHead(d_model, self.env.n_actions - 1)
        self.logF_head = TransformerLogFHead(d_model)
        
        # Create estimators with masked policies
        self.pf_estimator = DiscretePolicyEstimator(
            module=lambda x: self.compute_forward_policy(x),
            n_actions=self.env.n_actions,
            is_backward=False,
            preprocessor=self.env.preprocessor
        )
        
        self.pb_estimator = DiscretePolicyEstimator(
            module=lambda x: self.compute_backward_policy(x),
            n_actions=self.env.n_actions - 1,
            is_backward=True,
            preprocessor=self.env.preprocessor
        )
        
        self.logF_estimator = ScalarEstimator(
            module=self.compute_logF,
            preprocessor=self.env.preprocessor
        )
        
        # Create GFlowNet
        self.gflownet = SubTBGFlowNet(
            pf=self.pf_estimator,
            pb=self.pb_estimator,
            logF=self.logF_estimator,
            lamda=lambda_
        )
        
        self.sampler = Sampler(estimator=self.pf_estimator)
        
    def compute_forward_policy(self, states):
        """Compute forward policy with proper masking and numerical stability."""
        # Get the states tensor
        if hasattr(states, 'tensor'):
            states_tensor = states.tensor
        else:
            states_tensor = states

        # Identify sink states
        is_sink = torch.all(states_tensor == self.env.sf.to(dtype=states_tensor.dtype), dim=1)
        non_sink = ~is_sink

        # Initialize log_probs
        log_probs = torch.full((len(states), self.env.n_actions), -20.0, device=states_tensor.device)

        # For non-sink states
        if non_sink.any():
            features = self.transformer(states_tensor[non_sink])
            base_logits = self.pf_head(features)

            # Get masks
            masks = self.env.update_masks(states_tensor[non_sink])[0]

            # Apply temperature scaling
            if self.temperature != 1.0:
                base_logits = base_logits / self.temperature

            # Apply masking and compute log probabilities
            masked_logits = base_logits.masked_fill(~masks, -float('inf'))
            log_probs_subset = torch.log_softmax(masked_logits, dim=1)
            log_probs[non_sink] = log_probs_subset

        # For sink states, assign uniform probabilities
        if is_sink.any():
            n_actions = self.env.n_actions
            log_probs[is_sink] = -torch.log(torch.tensor(float(n_actions), device=states_tensor.device))

        return log_probs


    def compute_backward_policy(self, states):
        """Compute backward policy with proper masking and numerical stability."""
        # Get the states tensor
        if hasattr(states, 'tensor'):
            states_tensor = states.tensor
        else:
            states_tensor = states

        # Identify sink states
        is_sink = torch.all(states_tensor == self.env.sf.to(dtype=states_tensor.dtype), dim=1)
        non_sink = ~is_sink

        # Initialize log_probs
        log_probs = torch.zeros(len(states), self.env.n_actions - 1, device=states_tensor.device)

        # For non-sink states
        if non_sink.any():
            features = self.transformer(states_tensor[non_sink])
            base_logits = self.pb_head(features)

            # Get masks
            masks = self.env.update_masks(states_tensor[non_sink])[1]

        # Apply temperature scaling
        if self.temperature != 1.0:
            base_logits = base_logits / self.temperature

        # Apply masking and compute log probabilities
        masked_logits = base_logits.masked_fill(~masks, -float('inf'))
        log_probs_subset = torch.log_softmax(masked_logits, dim=1)
        log_probs[non_sink] = log_probs_subset

        # For sink states, assign uniform probabilities
        if is_sink.any():
            n_actions = self.env.n_actions - 1
            log_probs[is_sink] = -torch.log(torch.tensor(float(n_actions), device=states_tensor.device))

        return log_probs


        
    def compute_logF(self, states):
        """Compute log flow."""
        # Get the states tensor
        if hasattr(states, 'tensor'):
            states_tensor = states.tensor
        else:
            states_tensor = states

        # Identify sink states
        is_sink = torch.all(states_tensor == self.env.sf.to(dtype=states_tensor.dtype), dim=1)
        non_sink = ~is_sink

        logF = torch.zeros(len(states), device=states_tensor.device)

        # For non-sink states
        if non_sink.any():
            features = self.transformer(states_tensor[non_sink])
            logF[non_sink] = self.logF_head(features).squeeze(-1)

        # For sink states, set logF to the log reward
        if is_sink.any():
            logF[is_sink] = self.env.log_reward(states_tensor[is_sink])
        return logF

    
    def forward(self, x):
        """Forward pass (required for nn.Module)."""
        return self.transformer(x)
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.transformer = self.transformer.to(device)
        self.pf_head = self.pf_head.to(device)
        self.pb_head = self.pb_head.to(device)
        self.logF_head = self.logF_head.to(device)
        return self
    
    def sample_feature_sets(self, n_samples: int = 1) -> torch.Tensor:
        """Sample n different feature sets."""
        try:
            trajectories = self.sampler.sample_trajectories(self.env, n=n_samples)
            terminal_states = trajectories.states.tensor[trajectories.done]
            return terminal_states
        except Exception as e:
            print(f"Error during sampling: {e}")
            return None
    
    def get_optimizer(self, lr_transformer: float = 1e-4, lr_heads: float = 1e-3):
        """Create optimizer with separate learning rates."""
        return torch.optim.Adam([
            {'params': self.transformer.parameters(), 'lr': lr_transformer},
            {'params': self.pf_head.parameters(), 'lr': lr_heads},
            {'params': self.pb_head.parameters(), 'lr': lr_heads},
            {'params': self.logF_head.parameters(), 'lr': lr_heads}
        ])