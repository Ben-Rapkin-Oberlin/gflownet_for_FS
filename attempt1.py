import torch
import torch
from gfn.gflownet import GFlowNet, Trainer #RewardFunction, Trainer

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

# Load data
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
num_features = X.shape[1]
subset_size = 5  # Number of features to select

# Define the reward function
class FeatureSelectionReward():
    def __init__(self, X, y, estimator, cv=5):
        super().__init__()
        self.X = X
        self.y = y
        self.estimator = estimator
        self.cv = cv

    def forward(self, state):
        if len(state) != subset_size:
            return torch.tensor(0.0)
        X_subset = self.X[:, state]
        scores = cross_val_score(self.estimator, X_subset, self.y, cv=self.cv, scoring='accuracy')
        reward = np.mean(scores)
        return torch.tensor(reward)

# Initialize reward function
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
reward_fn = FeatureSelectionReward(X, y, estimator, cv=5)

# Define the GFlowNet
class FeatureSelectionGFlowNet(GFlowNet):
    def __init__(self, num_features, subset_size, reward_fn, hidden_size=128, num_layers=2):
        super().__init__()
        self.num_features = num_features
        self.subset_size = subset_size
        self.reward_fn = reward_fn
        # Define your neural network architecture here
        # This is a placeholder; define according to torchgfn requirements
        self.network = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, subset_size),
            torch.nn.Softmax(dim=-1)
        )

    def forward_policy(self, state):
        # Define forward policy based on current state
        current_size = len(state)
        if current_size >= self.subset_size:
            return []

        possible_actions = set(range(self.num_features)) - set(state)
        return list(possible_actions)

    def backward_policy(self, state):
        # Define backward policy based on current state
        if not state:
            return []
        predecessors = []
        for feature in state:
            predecessor = tuple(sorted(set(state) - {feature}))
            predecessors.append(predecessor)
        return predecessors

    def get_reward(self, state):
        if len(state) == self.subset_size:
            return self.reward_fn(state).item()
        return 0.0

    def is_terminal(self, state):
        return len(state) == self.subset_size

# Initialize GFlowNet
gfn = FeatureSelectionGFlowNet(
    num_features=num_features,
    subset_size=subset_size,
    reward_fn=reward_fn,
    hidden_size=128,
    num_layers=2
)

# Initialize Trainer
trainer = Trainer(
    model=gfn,
    reward_fn=reward_fn,
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=1000
)

# Train the GFlowNet
trainer.train()

# Sample feature subsets
num_samples = 10
samples = gfn.sample(num_samples)

# Decode and display sampled feature subsets
selected_features = [list(sample) for sample in samples]
for i, subset in enumerate(selected_features):
    feature_names_subset = feature_names[list(subset)]
    print(f"Sample {i+1}: {feature_names_subset}")