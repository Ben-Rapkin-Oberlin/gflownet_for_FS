import torch
import torch.nn as nn
from gfn.gflownet import SubTBGFlowNet
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.samplers import Sampler

class FeatureSelectionGFlowNet:
    """
    GFlowNet for feature selection using Sub Trajectory Balance with Transformer architecture.
    """
    def __init__(
        self,
        n_features: int,
        target_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        lambda_: float = 0.9
    ):
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
        
        # Create full modules by combining transformer with heads
        self.module_PF = nn.Module()
        self.module_PB = nn.Module()
        self.module_logF = nn.Module()
        
        # Define forward passes
        def make_forward(head):
            def forward(x):
                features = self.transformer(x)
                return head(features)
            return forward
            
        self.module_PF.forward = make_forward(self.pf_head)
        self.module_PB.forward = make_forward(self.pb_head)
        self.module_logF.forward = make_forward(self.logF_head)
        
        # Create estimators
        self.pf_estimator = DiscretePolicyEstimator(
            self.module_PF,
            self.env.n_actions,
            is_backward=False,
            preprocessor=self.env.preprocessor
        )
        
        self.pb_estimator = DiscretePolicyEstimator(
            self.module_PB,
            self.env.n_actions,
            is_backward=True,
            preprocessor=self.env.preprocessor
        )
        
        self.logF_estimator = ScalarEstimator(
            module=self.module_logF,
            preprocessor=self.env.preprocessor
        )
        
        # Create GFlowNet and sampler
        self.gflownet = SubTBGFlowNet(
            pf=self.pf_estimator,
            pb=self.pb_estimator,
            logF=self.logF_estimator,
            lamda=lambda_
        )
        self.sampler = Sampler(estimator=self.pf_estimator)
    
    def get_optimizer(self, lr_transformer: float = 1e-4, lr_heads: float = 1e-3):
        """
        Create optimizer with separate learning rates for transformer and heads.
        """
        # Create optimizer groups with different learning rates
        optimizer = torch.optim.Adam([
            {'params': self.transformer.parameters(), 'lr': lr_transformer},
            {'params': self.pf_head.parameters(), 'lr': lr_heads},
            {'params': self.pb_head.parameters(), 'lr': lr_heads},
            {'params': self.logF_head.parameters(), 'lr': lr_heads}
        ])
        
        return optimizer
    
    def sample_feature_sets(self, n_samples: int = 1) -> torch.Tensor:
        """Sample n different feature sets."""
        trajectories = self.sampler.sample_trajectories(self.env, n=n_samples)
        terminal_states = trajectories.states.tensor[trajectories.done]
        return terminal_states
    
    def get_parameters(self):
        """Get all trainable parameters of the model."""
        return {
            'transformer': self.transformer.state_dict(),
            'pf_head': self.pf_head.state_dict(),
            'pb_head': self.pb_head.state_dict(),
            'logF_head': self.logF_head.state_dict()
        }
    
    def load_parameters(self, parameters):
        """Load parameters into the model."""
        self.transformer.load_state_dict(parameters['transformer'])
        self.pf_head.load_state_dict(parameters['pf_head'])
        self.pb_head.load_state_dict(parameters['pb_head'])
        self.logF_head.load_state_dict(parameters['logF_head'])