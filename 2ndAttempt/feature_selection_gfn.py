import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple
from gfn.gflownet import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.utils.modules import MLP
from gfn.containers import Trajectories
from gfn.states import DiscreteStates, States

class FeaturePreprocessor:
    """Preprocessor for feature selection states."""
    def __init__(self, n_features):
        self.output_dim = n_features

    def __call__(self, states):
        """Convert states to tensor representation."""
        if isinstance(states, FeatureSelectionStates):
            return states.tensor
        return states

@dataclass
class FeatureSelectionState:
    """Represents the state in feature selection process."""
    selected_features: List[int]  # Indices of selected features
    available_features: List[int]  # Indices of features still available
    is_sink: bool = False
    
    def __eq__(self, other):
        if not isinstance(other, FeatureSelectionState):
            return False
        return (self.selected_features == other.selected_features and 
                self.available_features == other.available_features and
                self.is_sink == other.is_sink)

class FeatureSelectionStates(DiscreteStates):
    """Container for batches of feature selection states."""
    
    def __init__(self, states: List[FeatureSelectionState]):
        self.states = states
        self.batch_shape = (len(states),)
        self._tensor = None

    @property
    def tensor(self):
        """Convert states to tensor representation."""
        if self._tensor is None:
            n_features = len(self.states[0].available_features) + len(self.states[0].selected_features)
            self._tensor = torch.zeros((len(self.states), n_features))
            for i, state in enumerate(self.states):
                if not state.is_sink:
                    for idx in state.selected_features:
                        self._tensor[i, idx] = 1.0
        return self._tensor
    
    def __len__(self):
        return len(self.states)
    
    def extend(self, other):
        if isinstance(other, FeatureSelectionStates):
            self.states.extend(other.states)
            self.batch_shape = (len(self.states),)
            self._tensor = None
        else:
            raise ValueError("Can only extend with another FeatureSelectionStates object")

    @classmethod
    def from_states(cls, states):
        if isinstance(states, list):
            return cls(states)
        elif isinstance(states, FeatureSelectionState):
            return cls([states])
        else:
            raise ValueError(f"Cannot create {cls.__name__} from {type(states)}")
            
    @property
    def device(self):
        return self.tensor.device

    @property
    def is_sink_state(self):
        return torch.tensor([state.is_sink for state in self.states])

    @classmethod
    def get_sink_state(cls, n_features):
        return FeatureSelectionState(
            selected_features=[],
            available_features=[],
            is_sink=True
        )

    @property
    def shape(self):
        return self.tensor.shape[1:]  # Return shape excluding batch dimension

class FeatureSelectionEnv:
    """Environment for feature selection using GFlowNet."""
    
    def __init__(
        self, 
        n_features: int,
        target_size: int,
        reward_fn,
        device: str = "cpu"
    ):
        self.n_features = n_features
        self.target_size = target_size
        self.reward_fn = reward_fn
        self.device = device
        
        # Create preprocessor
        self.preprocessor = FeaturePreprocessor(n_features)
        
        # Initial state
        self.initial_state = FeatureSelectionState(
            selected_features=[],
            available_features=list(range(n_features))
        )
        
        # Sink state
        self.sink_state = FeatureSelectionState(
            selected_features=[],
            available_features=[],
            is_sink=True
        )
        
        # Number of possible actions
        self.n_actions = n_features + 1  # Add one for the sink state action

    def actions_from_batch_shape(self, batch_shape):
        """Create a tensor of actions of the given batch shape."""
        return torch.zeros(batch_shape + (1,), dtype=torch.long, device=self.device)
        
    def reset(self, batch_shape=()):
        """Reset the environment and return initial states."""
        if not batch_shape:
            return FeatureSelectionStates([self.initial_state])
            
        initial_states = [self.initial_state for _ in range(batch_shape[0])]
        return FeatureSelectionStates(initial_states)

    def step(self, states: FeatureSelectionStates, actions: torch.Tensor) -> Tuple[FeatureSelectionStates, torch.Tensor]:
        """Take a step in the environment."""
        next_states = []
        is_terminal = torch.zeros(len(states), dtype=torch.bool, device=self.device)
        
        for i, (state, action) in enumerate(zip(states.states, actions)):
            action_item = action.item()
            
            # Handle sink state transitions
            if action_item == self.n_features or state.is_sink:
                next_state = self.sink_state
                is_terminal[i] = True
            elif action_item in state.available_features:
                next_state = FeatureSelectionState(
                    selected_features=state.selected_features + [action_item],
                    available_features=[f for f in state.available_features if f != action_item]
                )
                if len(next_state.selected_features) == self.target_size:
                    is_terminal[i] = True
            else:
                next_state = state  # Invalid action, stay in current state
                
            next_states.append(next_state)
            
        return FeatureSelectionStates(next_states), is_terminal

    def log_reward(self, states: FeatureSelectionStates) -> torch.Tensor:
        """Compute log reward for given states."""
        rewards = torch.zeros(len(states), device=self.device)
        for i, state in enumerate(states.states):
            if len(state.selected_features) == self.target_size:
                rewards[i] = torch.tensor(self.reward_fn(state.selected_features), device=self.device)
        return torch.log(rewards + 1e-10)  # Add small constant to avoid log(0)

    def preprocess_states(self, states: FeatureSelectionStates) -> torch.Tensor:
        """Convert states to tensor representation."""
        return states.tensor.to(self.device)

# [Rest of the classes and functions remain the same]
@dataclass
class FeatureSelectionState:
    """Represents the state in feature selection process."""
    selected_features: List[int]  # Indices of selected features
    available_features: List[int]  # Indices of features still available
    is_sink: bool = False
    
    def __eq__(self, other):
        if not isinstance(other, FeatureSelectionState):
            return False
        return (self.selected_features == other.selected_features and 
                self.available_features == other.available_features and
                self.is_sink == other.is_sink)

class FeatureSelectionStates(DiscreteStates):
    """Container for batches of feature selection states."""
    
    def __init__(self, states: List[FeatureSelectionState]):
        self.states = states
        self.batch_shape = (len(states),)
        self._tensor = None

    @property
    def tensor(self):
        """Convert states to tensor representation."""
        if self._tensor is None:
            n_features = len(self.states[0].available_features) + len(self.states[0].selected_features)
            self._tensor = torch.zeros((len(self.states), n_features))
            for i, state in enumerate(self.states):
                if not state.is_sink:
                    for idx in state.selected_features:
                        self._tensor[i, idx] = 1.0
        return self._tensor
    
    def __len__(self):
        return len(self.states)
    
    def extend(self, other):
        if isinstance(other, FeatureSelectionStates):
            self.states.extend(other.states)
            self.batch_shape = (len(self.states),)
            self._tensor = None
        else:
            raise ValueError("Can only extend with another FeatureSelectionStates object")

    @classmethod
    def from_states(cls, states):
        if isinstance(states, list):
            return cls(states)
        elif isinstance(states, FeatureSelectionState):
            return cls([states])
        else:
            raise ValueError(f"Cannot create {cls.__name__} from {type(states)}")
            
    @property
    def device(self):
        return self.tensor.device

    @property
    def is_sink_state(self):
        return torch.tensor([state.is_sink for state in self.states])

    @classmethod
    def get_sink_state(cls, n_features):
        return FeatureSelectionState(
            selected_features=[],
            available_features=[],
            is_sink=True
        )

    @property
    def shape(self):
        return self.tensor.shape[1:]  # Return shape excluding batch dimension

class FeatureSelectionEnv:
    """Environment for feature selection using GFlowNet."""
    
    def __init__(
        self, 
        n_features: int,
        target_size: int,
        reward_fn,
        device: str = "cpu"
    ):
        self.n_features = n_features
        self.target_size = target_size
        self.reward_fn = reward_fn
        self.device = device
        
        # Initial state
        self.initial_state = FeatureSelectionState(
            selected_features=[],
            available_features=list(range(n_features))
        )
        
        # Sink state
        self.sink_state = FeatureSelectionState(
            selected_features=[],
            available_features=[],
            is_sink=True
        )
        
        # Number of possible actions
        self.n_actions = n_features + 1  # Add one for the sink state action

    def actions_from_batch_shape(self, batch_shape):
        """Create a tensor of actions of the given batch shape."""
        return torch.zeros(batch_shape + (1,), dtype=torch.long, device=self.device)
        
    def reset(self, batch_shape=()):
        """Reset the environment and return initial states."""
        if not batch_shape:
            return FeatureSelectionStates([self.initial_state])
            
        initial_states = [self.initial_state for _ in range(batch_shape[0])]
        return FeatureSelectionStates(initial_states)
class FeatureSelectionPolicy(nn.Module):
    """Policy network for feature selection."""
    
    def __init__(self, n_features: int, hidden_dim: int = 128):
        super().__init__()
        self.network = MLP(
            input_dim=n_features,
            output_dim=n_features + 1,  # Add one for sink state action
            hidden_dim=hidden_dim,
            n_hidden_layers=2,
            activation_fn="relu"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        mask = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1) > 0.5
        logits = logits.masked_fill(mask, float('-inf'))
        return logits

def create_feature_selection_gfn(env, device="cpu"):
    """Create a GFlowNet for feature selection."""
    
    # Create policy networks
    pf_module = FeatureSelectionPolicy(
        n_features=env.n_features,
        hidden_dim=128
    ).to(device)
    
    pb_module = FeatureSelectionPolicy(
        n_features=env.n_features,
        hidden_dim=128
    ).to(device)
    
    # Create estimators
    pf_estimator = DiscretePolicyEstimator(
        module=pf_module,
        n_actions=env.n_actions,
        is_backward=False,
        preprocessor=env.preprocessor
    )
    
    pb_estimator = DiscretePolicyEstimator(
        module=pb_module,
        n_actions=env.n_actions,
        is_backward=True,
        preprocessor=env.preprocessor
    )
    
    return TBGFlowNet(
        pf=pf_estimator,
        pb=pb_estimator,
        logZ=0.0
    ).to(device)

def train_feature_selection_gfn(
    gflownet,
    env,
    n_iterations=1000,
    batch_size=32,
    lr_policy=1e-3,
    lr_logz=1e-1,
    device="cpu"
):
    """Train the feature selection GFlowNet."""
    
    optimizer = torch.optim.Adam(gflownet.pf_pb_parameters(), lr=lr_policy)
    optimizer.add_param_group({
        "params": gflownet.logz_parameters(),
        "lr": lr_logz
    })
    
    for iteration in range(n_iterations):
        trajectories = gflownet.sample_trajectories(
            env=env,
            n=batch_size,
            save_logprobs=True
        )
        
        optimizer.zero_grad()
        loss = gflownet.loss(env, trajectories)
        loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.4f}")
    
    return gflownet