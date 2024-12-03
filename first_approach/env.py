import torch
import numpy as np
from typing import Tuple
from gfn.states import DiscreteStates
from gfn.env import DiscreteEnv

class FeatureSelectionEnv(DiscreteEnv):
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
        s0 = torch.zeros(n_features, dtype=torch.float32)
        
        # Define sink state (used for padding)
        sf = torch.full((n_features,), -1, dtype=torch.float32)
        
        # Define state shape and number of actions
        state_shape = (n_features,)
        n_actions = n_features + 1  # one per feature + exit action
        
        # Initialize base class with required parameters
        super().__init__(
            n_actions=n_actions,
            s0=s0,
            state_shape=state_shape,
            sf=sf
        )
        
        # Store other attributes
        self.action_shape = (1,)
        
    def make_states_class(self):
        """Create a States class with proper mask updating."""
        env = self
        
        class States(DiscreteStates):
            # Set class variables
            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            n_actions = env.n_actions
            device = env.s0.device  # Use device from s0 tensor

            def update_masks(self):
                self.forward_masks, self.backward_masks = env.update_masks(self.tensor)
                
        return States

    def update_masks(self, states) -> Tuple[torch.Tensor, torch.Tensor]:
                """Update forward and backward masks based on current states."""
                # Get the states tensor
                if hasattr(states, 'tensor'):
                        states_tensor = states.tensor
                else:
                        states_tensor = states

                batch_size = states_tensor.shape[0]
                
                # Initialize masks
                forward_masks = torch.zeros(batch_size, self.n_actions, dtype=torch.bool, device=states_tensor.device)
                backward_masks = torch.zeros(batch_size, self.n_actions - 1, dtype=torch.bool, device=states_tensor.device)
                
                # Count selected features
                n_selected = torch.sum(states_tensor == 1, dim=1)
                
                # Identify different state types
                is_sink = torch.all(states_tensor == self.sf.to(dtype=states_tensor.dtype), dim=1)
                can_select = ~is_sink & (n_selected < self.target_features)
                can_exit = ~is_sink & (n_selected == self.target_features)
                
                # Set feature selection masks
                if can_select.any():
                        forward_masks[can_select, :-1] = states_tensor[can_select] != 1
                
                # Set exit masks
                if can_exit.any():
                        forward_masks[can_exit, -1] = True
                
                # Set backward masks
                if (~is_sink).any():
                        backward_masks[~is_sink] = states_tensor[~is_sink] == 1
                
                # Debug info without can_exit reference
                n_sink = is_sink.sum().item()
                n_can_select = can_select.sum().item()
                n_can_exit = can_exit.sum().item()
                print(f"\nMasks: Sink={n_sink}, Can_select={n_can_select}, Can_exit={n_can_exit}")
                
                return forward_masks, backward_masks

    def step(self, states, actions) -> torch.Tensor:
                """Forward step in the environment."""
                # Get the states tensor
                if hasattr(states, 'tensor'):
                        states_tensor = states.tensor
                else:
                        states_tensor = states
                        
                next_states = states_tensor.clone()
                is_sink = torch.all(states_tensor == self.sf.to(dtype=states_tensor.dtype), dim=1)
                
                # No transitions from sink states
                if is_sink.all():
                        return next_states
                
                # Get the actions tensor and ensure it's 2D
                if hasattr(actions, 'tensor'):
                        actions_tensor = actions.tensor
                else:
                        actions_tensor = actions
                
                if actions_tensor.dim() == 1:
                        actions_tensor = actions_tensor.unsqueeze(-1)
                
                # Count features and identify transitioning states
                n_selected = torch.sum(next_states == 1, dim=1)
                at_target = n_selected == self.target_features
                
                print("\nStep Debug:")
                print(f"Current features: {n_selected[0].item()}")
                print(f"States at target: {at_target.sum().item()}")
                print(f"Sink states: {is_sink.sum().item()}")
                
                # Handle states that should transition to sink
                if at_target.any():
                        next_states[at_target] = self.sf
                        print(f"Transitioned {at_target.sum().item()} states to sink")
                
                # Handle remaining feature selection
                can_select = ~at_target & ~is_sink & (n_selected < self.target_features)
                if can_select.any():
                        valid_actions = torch.clamp(actions_tensor[can_select, 0], 0, self.n_features - 1)
                        batch_indices = torch.arange(len(next_states), device=next_states.device)[can_select]
                        next_states[batch_indices, valid_actions] = 1
                        print(f"Selected features for {can_select.sum().item()} states")
                
                return next_states

    def _debug_masks(self, states_tensor, forward_masks, backward_masks):
                """Debug helper to analyze mask properties."""
                n_selected = torch.sum(states_tensor == 1, dim=1)
                is_sink = torch.all(states_tensor == self.sf.to(dtype=states_tensor.dtype), dim=1)
                
                print("\nDEBUG: Environment mask analysis:")
                print(f"States shape: {states_tensor.shape}")
                print(f"Forward masks shape: {forward_masks.shape}")
                print(f"Backward masks shape: {backward_masks.shape}")
                print("\nPer-state analysis:")
                for i in range(len(states_tensor)):
                    print(f"\nState {i}:")
                    print(f"  Features selected: {n_selected[i].item()}")
                    print(f"  Is sink: {is_sink[i].item()}")
                    print(f"  Valid forward actions: {forward_masks[i].sum().item()}")
                    print(f"  Exit allowed: {forward_masks[i, -1].item()}")
                    print(f"  Valid backward actions: {backward_masks[i].sum().item()}")
                    if n_selected[i] >= self.target_features and not forward_masks[i, -1]:
                        print("  WARNING: State at/above target features but exit not allowed!")
                    if forward_masks[i].sum() == 0:
                        print("  WARNING: No valid forward actions!")


	
    def backward_step(self, states, actions) -> torch.Tensor:
            """Backward step in the environment."""
            # Get the states tensor
            if hasattr(states, 'tensor'):
                states_tensor = states.tensor
            else:
                states_tensor = states
                
            prev_states = states_tensor.clone()
            
            # Get the actions tensor
            if hasattr(actions, 'tensor'):
                actions_tensor = actions.tensor
            else:
                actions_tensor = actions
                
            # Don't modify sink states
            is_sink = torch.all(states_tensor == self.sf.to(dtype=states_tensor.dtype), dim=1)
            if is_sink.all():
                return prev_states
                
            # Ensure actions are 2D
            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(-1)
                
            # Handle feature deselection for non-sink states
            non_sink_mask = ~is_sink
            if non_sink_mask.any():
                feature_indices = actions_tensor[non_sink_mask, 0].long()
                batch_indices = torch.arange(len(prev_states), device=prev_states.device)[non_sink_mask]
                prev_states[batch_indices, feature_indices] = 0
                
            return prev_states

    def cleanup_trajectories(self, states):
                """Ensure trajectories remain valid after transitions."""
                if hasattr(states, 'tensor'):
                        states_tensor = states.tensor
                else:
                        states_tensor = states
                
                # Handle sink states
                is_sink = torch.all(states_tensor == self.sf.to(dtype=states_tensor.dtype), dim=1)
                
                # Initialize masks if they don't exist
                if not hasattr(states, 'forward_masks'):
                    self.forward_masks = torch.zeros(states_tensor.shape[0], self.n_actions, 
                                                   dtype=torch.bool, device=states_tensor.device)
                if not hasattr(states, 'backward_masks'):
                    self.backward_masks = torch.zeros(states_tensor.shape[0], self.n_actions - 1, 
                                                    dtype=torch.bool, device=states_tensor.device)
                
                # For non-sink states, update masks normally
                if (~is_sink).any():
                    fwd_masks, back_masks = self.update_masks(states_tensor[~is_sink])
                    self.forward_masks[~is_sink] = fwd_masks
                    self.backward_masks[~is_sink] = back_masks
                
                # For sink states, set special mask values
                if is_sink.any():
                    # Sink states can't take any forward actions
                    self.forward_masks[is_sink] = False
                    
                    # For sink states, set uniform backward probability over all possible features
                    self.backward_masks[is_sink] = torch.ones(self.n_actions - 1, 
                                                            dtype=torch.bool, 
                                                            device=states_tensor.device)
                
                # Verify masks and print debug info
                print("\nCleanup Trajectories Debug:")
                print(f"Total states: {len(states_tensor)}")
                print(f"Sink states: {is_sink.sum().item()}")
                print(f"Non-sink states: {(~is_sink).sum().item()}")
                print(f"Forward mask sums: {self.forward_masks.sum(dim=1).tolist()}")
                print(f"Backward mask sums: {self.backward_masks.sum(dim=1).tolist()}")
                
                # Verify masks
                assert self.forward_masks.shape[1] == self.n_actions, \
                    f"Invalid forward mask shape: {self.forward_masks.shape}"
                assert self.backward_masks.shape[1] == self.n_actions - 1, \
                    f"Invalid backward mask shape: {self.backward_masks.shape}"
                
                # Print details for first few states
                print("\nDetailed mask information for first 3 states:")
                for i in range(min(3, len(states_tensor))):
                    print(f"\nState {i}:")
                    print(f"Is sink: {is_sink[i].item()}")
                    print(f"Forward mask: {self.forward_masks[i].tolist()}")
                    print(f"Backward mask: {self.backward_masks[i].tolist()}")
                
                # Add masks to states object if it exists
                if hasattr(states, 'forward_masks'):
                    states.forward_masks = self.forward_masks
                if hasattr(states, 'backward_masks'):
                    states.backward_masks = self.backward_masks

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

    def is_terminal_state(self, states: torch.Tensor) -> torch.Tensor:
        """Check if states are terminal (have exactly target_features selected)."""
        n_selected = torch.sum(states == 1, dim=1)
        return n_selected == self.target_features

    def log_reward(self, final_states: torch.Tensor) -> torch.Tensor:
        """Calculate log reward for terminal states."""
        # Placeholder - return uniform reward for all valid terminal states
        is_valid = self.is_terminal_state(final_states)
        return torch.where(is_valid, 
                         torch.tensor(0.0),
                         torch.tensor(float('-inf')))
    
    def reset(self, batch_shape=(1,)):
        """Reset the environment and return initial states."""
        States = self.make_states_class()
        # Create initial tensor with proper batch shape
        initial_tensor = self.s0.expand(*batch_shape, -1).clone()
        # Initialize states with just the tensor - masks will be created automatically
        states = States(tensor=initial_tensor)
        # Update the masks based on the current states
        states.update_masks()
        return states
