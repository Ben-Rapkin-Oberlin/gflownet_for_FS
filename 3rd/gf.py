import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class GFlowNet:
    def __init__(self, state_dim, hidden_dim=150, num_layers=3):
        """
        Initialize a GFlowNet.
        
        Args:
            state_dim (int): Dimension of the state space
            hidden_dim (int): Number of neurons in hidden layers
            num_layers (int): Number of layers in the neural network
        """
        self.state_dim = state_dim
        
        # Initialize the flow network (using the paper's architecture)
        layers = []
        input_dim = state_dim
        
        # First layer with Fourier features as suggested in the paper
        self.fourier_scale = 0.1  # From paper's hyperparameters
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Sine())  # Sine activation for Fourier features
        
        # Additional layers with ReLU activation
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        # Output layer for flow values
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # Ensure positive flow values
        
        self.flow_network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.flow_network.parameters(), lr=2e-4)

    def compute_flow(self, state):
        """Compute the flow value for a given state."""
        with torch.no_grad():
            return self.flow_network(state)

    def forward_policy(self, state, possible_actions):
        """
        Compute forward policy probabilities P_F(s'|s) using flow values.
        Returns action probabilities according to eq. (6) in the paper.
        """
        flows = []
        for action in possible_actions:
            next_state = self.apply_action(state, action)
            flow = self.compute_flow(next_state)
            flows.append(flow)
            
        flows = torch.tensor(flows)
        # Normalize to get probabilities
        probs = flows / flows.sum()
        return probs

    def sample_trajectory(self, initial_state, max_steps, exploration_prob=0.1):
        """
        Sample a trajectory through the state space using the learned policy.
        
        Args:
            initial_state: Starting state
            max_steps: Maximum number of steps to take
            exploration_prob: Probability of taking a random action
        
        Returns:
            List of (state, action) pairs forming a trajectory
        """
        trajectory = []
        current_state = initial_state
        
        for _ in range(max_steps):
            possible_actions = self.get_valid_actions(current_state)
            if not possible_actions:
                break
                
            if np.random.random() < exploration_prob:
                # Random exploration
                action = np.random.choice(possible_actions)
            else:
                # Sample from policy
                probs = self.forward_policy(current_state, possible_actions)
                action_idx = torch.multinomial(probs, 1).item()
                action = possible_actions[action_idx]
            
            next_state = self.apply_action(current_state, action)
            trajectory.append((current_state, action))
            current_state = next_state
            
        return trajectory

    def train_step(self, batch_trajectories, rewards):
        """
        Perform one training step using the flow-matching objective (eq. 9 in paper).
        
        Args:
            batch_trajectories: List of trajectories
            rewards: Corresponding rewards for terminal states
        """
        self.optimizer.zero_grad()
        loss = 0
        
        for trajectory, reward in zip(batch_trajectories, rewards):
            # Implement flow matching loss from equation (9)
            for state, next_state in zip(trajectory[:-1], trajectory[1:]):
                incoming_flow = self.compute_flow(state)
                outgoing_flow = self.compute_flow(next_state)
                
                # Flow matching objective
                flow_diff = incoming_flow - (reward if next_state[-1] else 0) - outgoing_flow
                loss += flow_diff ** 2
                
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def get_valid_actions(self, state):
        """
        Get list of valid actions from current state.
        To be implemented based on specific problem constraints.
        """
        raise NotImplementedError

    def apply_action(self, state, action):
        """
        Apply action to state to get next state.
        To be implemented based on specific problem.
        """
        raise NotImplementedError

class SetSelectionGFlowNet(GFlowNet):
    """
    Specialized GFlowNet for set selection problems as described in the paper.
    """
    def __init__(self, num_elements, target_size, hidden_dim=150, num_layers=3):
        """
        Initialize a GFlowNet for set selection.
        
        Args:
            num_elements (int): Total number of elements to select from
            target_size (int): Size of target subset to select
            hidden_dim (int): Number of neurons in hidden layers
            num_layers (int): Number of layers in the neural network
        """
        super().__init__(num_elements, hidden_dim, num_layers)
        self.num_elements = num_elements
        self.target_size = target_size

    def get_valid_actions(self, state):
        """Get valid elements that can be added to current subset."""
        current_state = torch.tensor(state)
        # Can add any element that isn't already in the set
        # and only if we haven't reached target size
        if current_state.sum() >= self.target_size:
            return []
        return [i for i, x in enumerate(state) if x == 0]

    def apply_action(self, state, action):
        """Add selected element to subset."""
        new_state = state.clone()
        new_state[action] = 1
        return new_state

    def sample_subset(self, reward_func, num_samples=1):
        """
        Sample subsets using the trained GFlowNet.
        
        Args:
            reward_func: Function that computes reward for a subset
            num_samples: Number of subsets to sample
            
        Returns:
            List of sampled subsets
        """
        subsets = []
        for _ in range(num_samples):
            initial_state = torch.zeros(self.num_elements)
            trajectory = self.sample_trajectory(initial_state, self.target_size)
            # Extract final subset from trajectory
            subset = torch.zeros(self.num_elements)
            for _, action in trajectory:
                subset[action] = 1
            subsets.append(subset)
        return subsets