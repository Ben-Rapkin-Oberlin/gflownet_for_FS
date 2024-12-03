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
                features = self.transformer(states)
                base_logits = self.pf_head(features)
                
                # Get states tensor and masks
                if hasattr(states, 'tensor'):
                        states_tensor = states.tensor
                else:
                        states_tensor = states
                
                if hasattr(states, 'forward_masks'):
                        masks = states.forward_masks
                else:
                        masks = self.env.update_masks(states)[0]
                
                # Get state information
                n_selected = torch.sum(states_tensor == 1, dim=1)
                is_sink = torch.all(states_tensor == self.env.sf.to(dtype=states_tensor.dtype), dim=1)
                
                # Initialize log probabilities with a reasonable negative value
                MIN_PROB = -20.0  # log(2e-9)
                log_probs = torch.full_like(base_logits, MIN_PROB)
                
                # For sink states, assign uniform probability to all actions
                if is_sink.any():
                        n_actions = base_logits.shape[1]
                        uniform_prob = -torch.log(torch.tensor(float(n_actions)))
                        log_probs[is_sink] = uniform_prob
                
                # Handle non-sink states
                non_sink = ~is_sink
                if non_sink.any():
                        valid_logits = base_logits[non_sink]
                        valid_masks = masks[non_sink]
                        
                        # Apply temperature scaling
                        if self.temperature != 1.0:
                            valid_logits = valid_logits / self.temperature
                        
                        # Apply masking
                        masked_logits = torch.where(
                                valid_masks,
                                valid_logits,
                                torch.tensor(MIN_PROB, device=valid_logits.device)
                        )
                        
                        # Compute log probabilities with numerical stability
                        max_logits = torch.max(masked_logits, dim=1, keepdim=True)[0]
                        exp_logits = torch.exp(torch.clamp(masked_logits - max_logits, min=MIN_PROB))
                        exp_logits = torch.where(valid_masks, exp_logits, torch.zeros_like(exp_logits))
                        sum_exp = torch.sum(exp_logits, dim=1, keepdim=True).clamp(min=1e-8)
                        valid_log_probs = masked_logits - max_logits - torch.log(sum_exp)
                        
                        # Ensure no invalid values
                        valid_log_probs = torch.where(
                            valid_masks,
                            torch.clamp(valid_log_probs, min=MIN_PROB),
                            torch.full_like(valid_log_probs, MIN_PROB)
                        )
                        
                        # Insert back into full tensor
                        log_probs[non_sink] = valid_log_probs
                
                # Print debug info
                with torch.no_grad():
                    probs = torch.exp(log_probs)
                    total_probs = probs.sum(dim=1)
                    print("\nForward Policy:")
                    print(f"Sink states: {is_sink.sum().item()}")
                    print(f"States at target: {(n_selected == self.env.target_features).sum().item()}")
                    print(f"Log probs range: [{log_probs.max().item():.4f}, {log_probs.min().item():.4f}]")
                    print(f"Total probability range: [{total_probs.min().item():.6f}, {total_probs.max().item():.6f}]")
                
                return log_probs

    def compute_backward_policy(self, states):
                """Compute backward policy with proper masking and numerical stability."""
                features = self.transformer(states)
                base_logits = self.pb_head(features)
                
                # Get states tensor and masks
                if hasattr(states, 'tensor'):
                        states_tensor = states.tensor
                else:
                        states_tensor = states
                
                # Get masks and ensure they're properly initialized
                if hasattr(states, 'backward_masks'):
                        masks = states.backward_masks
                else:
                        _, masks = self.env.update_masks(states_tensor)
                
                # Get state information
                is_sink = torch.all(states_tensor == self.env.sf.to(dtype=states_tensor.dtype), dim=1)
                
                # Initialize log probabilities with MIN_PROB
                MIN_PROB = -20.0
                n_actions = base_logits.shape[1]
                uniform_log_prob = -torch.log(torch.tensor(float(n_actions), device=base_logits.device))
                log_probs = torch.full_like(base_logits, MIN_PROB, device=base_logits.device)
                
                # Handle sink states by assigning uniform probabilities over valid actions
                if is_sink.any():
                    log_probs[is_sink] = uniform_log_prob
                
                # Handle non-sink states
                non_sink = ~is_sink
                if non_sink.any():
                    valid_logits = base_logits[non_sink]
                    valid_masks = masks[non_sink]
                    
                    # Only process states with valid backward actions
                    has_valid = valid_masks.sum(dim=1) > 0
                    if has_valid.any():
                        # Apply temperature scaling
                        scaled_logits = valid_logits / self.temperature
                        
                        # Compute masked log probabilities with numerical stability
                        masked_logits = scaled_logits.masked_fill(~valid_masks, float('-inf'))
                        log_probs_valid = torch.log_softmax(masked_logits, dim=1)
                        
                        # Replace -inf with MIN_PROB for invalid actions
                        log_probs_valid = torch.where(
                            valid_masks,
                            log_probs_valid,
                            torch.tensor(MIN_PROB, device=log_probs_valid.device)
                        )
                        
                        # Update log probabilities for non-sink states with valid actions
                        valid_indices = torch.where(non_sink)[0][has_valid]
                        log_probs[valid_indices] = log_probs_valid[has_valid]
                    
                    # For states without valid backward actions, use uniform distribution
                    invalid_indices = torch.where(non_sink)[0][~has_valid]
                    if len(invalid_indices) > 0:
                        log_probs[invalid_indices] = uniform_log_prob
                
                # Extended debug information
                with torch.no_grad():
                    probs = torch.exp(log_probs)
                    total_prob = probs.sum(dim=1)
                    print("\nBackward Policy Detailed Debug:")
                    print(f"Sink states: {is_sink.sum().item()}")
                    print(f"Min log prob: {log_probs.min().item():.4f}")
                    print(f"Max log prob: {log_probs.max().item():.4f}")
                    print(f"Total probability range: [{total_prob.min().item():.6f}, {total_prob.max().item():.6f}]")
                    print(f"Backward masks sum: {masks.sum(dim=1).tolist()}")
                    
                    if is_sink.any():
                        print("\nSink State Details:")
                        sink_indices = torch.where(is_sink)[0]
                        for idx in sink_indices[:5]:  # Show first 5 sink states
                            print(f"\nSink state {idx}:")
                            print(f"Logits: {base_logits[idx].tolist()}")
                            print(f"Probs: {probs[idx].tolist()}")
                            print(f"Mask: {masks[idx].tolist()}")
                    
                    # Detailed probability analysis
                    print("\nProbability Analysis:")
                    valid_masks = masks.sum(dim=1) > 0
                    print(f"States with valid backward actions: {valid_masks.sum().item()}")
                    print(f"States with no valid backward actions: {(~valid_masks).sum().item()}")
                    
                    # Check for potential numerical issues
                    print("\nNumerical Checks:")
                    print(f"NaN in logits: {torch.isnan(base_logits).any().item()}")
                    print(f"Inf in logits: {torch.isinf(base_logits).any().item()}")
                    print(f"NaN in log_probs: {torch.isnan(log_probs).any().item()}")
                    print(f"Inf in log_probs: {torch.isinf(log_probs).any().item()}")
                    print(f"NaN in probs: {torch.isnan(probs).any().item()}")
                    print(f"Inf in probs: {torch.isinf(probs).any().item()}")
                
                # Verify probability sum is approximately 1
                with torch.no_grad():
                    total_prob = torch.exp(log_probs).sum(dim=1)
                    assert torch.allclose(total_prob, torch.ones_like(total_prob), rtol=1e-4, atol=1e-4), \
                        f"Probabilities don't sum to 1: min={total_prob.min().item()}, max={total_prob.max().item()}"
                
                return log_probs
        
    def compute_logF(self, states):
        """Compute log flow."""
        features = self.transformer(states)
        return self.logF_head(features)
    
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