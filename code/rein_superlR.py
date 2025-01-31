#############V1: symplyfy##############
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# PyTorch for DNN
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler

# ---------------------------------------------
# 1) Generate Synthetic Data with Mild Correlation & ~0.80 AUC
# ---------------------------------------------
def generate_data(n_samples=10000, n_features=5, random_state=42):
    """
    Create a synthetic binary classification dataset with:
      - 2 truly informative features
      - 3 redundant features
      - 2 clusters per class
      - 20% label noise (flip_y=0.2)
      - class_sep=0.4 for partial overlap
      - mild correlation among features (~0.2)
    Expect best-model AUC ~0.75–0.85 (depending on random seed).
    """
    # 1) Generate with n_informative=2, n_clusters_per_class=2
    #    => avoids the "ValueError: must be <= 2**n_informative" issue
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,       # increased to 2 so 2*2 <= 2^2=4 is valid
        n_redundant=3,
        n_repeated=0,
        n_clusters_per_class=2,
        class_sep=0.4,         # smaller => more overlap
        flip_y=0.2,            # 20% label noise => lower AUC
        random_state=random_state
    )

    # 2) Introduce mild correlation among the 5 features
    corr = np.array([
        [1.0, 0.2, 0.2, 0.2, 0.2],
        [0.2, 1.0, 0.2, 0.2, 0.2],
        [0.2, 0.2, 1.0, 0.2, 0.2],
        [0.2, 0.2, 0.2, 1.0, 0.2],
        [0.2, 0.2, 0.2, 0.2, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    X = X.dot(L)

    # 3) Scale features
    X = StandardScaler().fit_transform(X)

    # Convert to DataFrame
    df_X = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(n_features)])
    s_y = pd.Series(y, name='Y')
    return df_X, s_y

# ---------------------------------------------
# 2) Metrics (AUC, KS)
# ---------------------------------------------
def calc_ks_score(y_true, y_prob):
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).sort_values('y_prob', ascending=False)
    data['cum_pos'] = (data['y_true'] == 1).cumsum()
    data['cum_neg'] = (data['y_true'] == 0).cumsum()
    total_pos = data['y_true'].sum()
    total_neg = (data['y_true'] == 0).sum()
    data['cum_pos_rate'] = data['cum_pos'] / total_pos
    data['cum_neg_rate'] = data['cum_neg'] / total_neg
    data['ks'] = data['cum_pos_rate'] - data['cum_neg_rate']
    return data['ks'].max()

# ---------------------------------------------
# 3) PyTorch DNN Model
# ---------------------------------------------
class DNNModel(nn.Module):
    def __init__(self, input_dim=5):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

def train_eval_pytorch_dnn(X_train, y_train, X_val, y_val,
                           epochs=5, batch_size=64, lr=1e-3, device='cpu'):
    model = DNNModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    dataset_size = len(X_train_t)
    n_batches = (dataset_size // batch_size) + 1

    for epoch in range(epochs):
        # Shuffle training data
        indices = torch.randperm(dataset_size)
        X_train_t = X_train_t[indices]
        y_train_t = y_train_t[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            if start_idx >= dataset_size:
                break

            x_batch = X_train_t[start_idx:end_idx]
            y_batch = y_train_t[start_idx:end_idx]

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy().ravel()

    auc = roc_auc_score(y_val, val_preds)
    ks = calc_ks_score(y_val, val_preds)
    return model, auc, ks, val_preds

# ---------------------------------------------
# 4) Helper: Train & Evaluate Various Models
# ---------------------------------------------
def train_eval_model(model_name, X_train, y_train, X_val, y_val, device='cpu'):
    if model_name == 'xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'lgbm':
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'rf':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'dnn':
        model, auc, ks, y_prob = train_eval_pytorch_dnn(
            X_train.values, y_train.values, X_val.values, y_val.values, device=device
        )
        return model, auc, ks, y_prob

    else:
        raise ValueError(f"Unknown model name: {model_name}")

# ---------------------------------------------
# 5) Weighted Blending
# ---------------------------------------------
def blend_predictions(probs_list, weights=None):
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    final_prob = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        final_prob += w * p
    return final_prob

def evaluate_action(action, X_train, X_val, y_train, y_val, device='cpu'):
    """
    action: int from 0..4 => (xgb=0, lgbm=1, rf=2, dnn=3, blend=4)
    Returns:
      reward = (auc + ks) - penalty
      auc, ks
    """
    model_names = ['xgb', 'lgbm', 'rf', 'dnn']
    if action < 4:
        chosen_model = model_names[action]
        _, auc_val, ks_val, _ = train_eval_model(chosen_model, X_train, y_train, X_val, y_val, device=device)
        penalty = 0.05 if chosen_model == 'dnn' else 0.0
        reward = (auc_val + ks_val) - penalty
        return reward, auc_val, ks_val
    else:
        # Blend
        probs_list = []
        for m in model_names:
            _, auc_m, ks_m, p = train_eval_model(m, X_train, y_train, X_val, y_val, device=device)
            probs_list.append(p)
        final_prob = blend_predictions(probs_list)
        auc_blend = roc_auc_score(y_val, final_prob)
        ks_blend = calc_ks_score(y_val, final_prob)
        reward = (auc_blend + ks_blend) - 0.1
        return reward, auc_blend, ks_blend

# ---------------------------------------------
# 6) A Simple Multi-Armed Bandit Approach
# ---------------------------------------------
def multi_armed_bandit_model_selection(
    n_episodes=50,
    n_actions=5,
    epsilon=0.06,
    device='cpu'
):
    """
    We have 5 actions (xgb=0, lgbm=1, rf=2, dnn=3, blend=4).
    For each 'episode':
      1) Generate a dataset (X,y) with the chosen seed
      2) Split into train/val
      3) Epsilon-greedy select an action
      4) Evaluate the chosen action => get reward
      5) Update average reward (Q) for that action
    """
    Q = np.zeros(n_actions, dtype=np.float32)
    counts = np.zeros(n_actions, dtype=int)

    # For storing raw AUC, KS, Reward each time an action is chosen
    action_auc_records = [[] for _ in range(n_actions)]
    action_ks_records = [[] for _ in range(n_actions)]
    action_reward_records = [[] for _ in range(n_actions)]

    action_history = []
    reward_history = []

    for episode in range(n_episodes):
        # Generate the data here
        seed = 1000 + episode
        X, y = generate_data(n_samples=5000, n_features=5, random_state=seed)

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123)

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q)

        # Evaluate chosen action => get (reward, auc, ks)
        reward, auc_val, ks_val = evaluate_action(
            action, X_train, X_val, y_train, y_val, device=device
        )

        # Update Q (incremental mean)
        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]

        # Store details
        action_history.append(action)
        reward_history.append(reward)
        action_auc_records[action].append(auc_val)
        action_ks_records[action].append(ks_val)
        action_reward_records[action].append(reward)

        print(f"Episode {episode+1}/{n_episodes}, "
              f"Action={action}, Reward={reward:.4f}, Updated Q={Q}")

    return Q, action_history, reward_history, action_auc_records, action_ks_records, action_reward_records

# ---------------------------------------------
# 7) Run the Bandit, then Interpret Results
# ---------------------------------------------
def run_bandit():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device={device}")

    n_episodes = 50
    n_actions = 5
    epsilon = 0.05

    (
        Q,
        actions,
        rewards,
        auc_records,
        ks_records,
        reward_records
    ) = multi_armed_bandit_model_selection(
        n_episodes=n_episodes,
        n_actions=n_actions,
        epsilon=epsilon,
        device=device
    )

    best_action = np.argmax(Q)
    model_names = ["XGB", "LightGBM", "RandomForest", "DNN", "Blend"]

    print("\n========================================")
    print("Interpreting Your Current Results")
    print("========================================\n")

    print("Final Q-values:", Q)
    print(f"Best action index: {best_action}")
    print(f"Best action is: {model_names[best_action]} with estimated Q = {Q[best_action]:.4f}\n")

    print("Detailed AUC/KS/Reward by action:")
    print("--------------------------------------------------")
    for a in range(n_actions):
        if len(auc_records[a]) > 0:
            avg_auc = np.mean(auc_records[a])
            avg_ks = np.mean(ks_records[a])
            avg_reward = np.mean(reward_records[a])
            print(f"Action {a} ({model_names[a]}): chosen {len(auc_records[a])} times")
            print(f"  Mean AUC = {avg_auc:.4f}, Mean KS = {avg_ks:.4f}, Mean Reward = {avg_reward:.4f}\n")
        else:
            print(f"Action {a} ({model_names[a]}): chosen 0 times\n")


if __name__ == "__main__":
    run_bandit()


#######V2: gym##################
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Gymnasium
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# For callback
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------
# 1) Generate Synthetic Data
# ---------------------------------------------
# ---------------------------------------------
# 1) Generate Synthetic Data with Mild Correlation & ~0.80 AUC
# ---------------------------------------------
def generate_data(n_samples=10000, n_features=5, random_state=42):
    """
    Create a synthetic binary classification dataset with:
      - 2 truly informative features
      - 3 redundant features
      - 2 clusters per class
      - 20% label noise (flip_y=0.2)
      - class_sep=0.4 for partial overlap
      - mild correlation among features (~0.2)
    Expect best-model AUC ~0.75–0.85 (depending on random seed).
    """
    # 1) Generate with n_informative=2, n_clusters_per_class=2
    #    => avoids the "ValueError: must be <= 2**n_informative" issue
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,       # increased to 2 so 2*2 <= 2^2=4 is valid
        n_redundant=3,
        n_repeated=0,
        n_clusters_per_class=2,
        class_sep=0.4,         # smaller => more overlap
        flip_y=0.2,            # 20% label noise => lower AUC
        random_state=random_state
    )

    # 2) Introduce mild correlation among the 5 features
    corr = np.array([
        [1.0, 0.2, 0.2, 0.2, 0.2],
        [0.2, 1.0, 0.2, 0.2, 0.2],
        [0.2, 0.2, 1.0, 0.2, 0.2],
        [0.2, 0.2, 0.2, 1.0, 0.2],
        [0.2, 0.2, 0.2, 0.2, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    X = X.dot(L)

    # 3) Scale features
    X = StandardScaler().fit_transform(X)

    # Convert to DataFrame
    df_X = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(n_features)])
    s_y = pd.Series(y, name='Y')
    return df_X, s_y

# ---------------------------------------------
# 2) Metrics (AUC, KS)
# ---------------------------------------------
def calc_ks_score(y_true, y_prob):
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).sort_values('y_prob', ascending=False)
    data['cum_pos'] = (data['y_true'] == 1).cumsum()
    data['cum_neg'] = (data['y_true'] == 0).cumsum()
    total_pos = data['y_true'].sum()
    total_neg = (data['y_true'] == 0).sum()
    data['cum_pos_rate'] = data['cum_pos'] / total_pos
    data['cum_neg_rate'] = data['cum_neg'] / total_neg
    data['ks'] = data['cum_pos_rate'] - data['cum_neg_rate']
    return data['ks'].max()


# ---------------------------------------------
# 3) PyTorch DNN
# ---------------------------------------------
class DNNModel(nn.Module):
    def __init__(self, input_dim=5):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x


def train_eval_pytorch_dnn(X_train, y_train, X_val, y_val,
                           epochs=5, batch_size=64, lr=1e-3, device='cpu'):
    model = DNNModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    dataset_size = len(X_train_t)
    n_batches = (dataset_size // batch_size) + 1

    for epoch in range(epochs):
        indices = torch.randperm(dataset_size)
        X_train_t = X_train_t[indices]
        y_train_t = y_train_t[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            if start_idx >= dataset_size:
                break

            x_batch = X_train_t[start_idx:end_idx]
            y_batch = y_train_t[start_idx:end_idx]

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy().ravel()

    auc = roc_auc_score(y_val, val_preds)
    ks = calc_ks_score(y_val, val_preds)
    return model, auc, ks, val_preds


# ---------------------------------------------
# 4) Train & Evaluate Helper
# ---------------------------------------------
def train_eval_model(model_name, X_train, y_train, X_val, y_val, device='cpu'):
    """
    Returns (model, auc, ks, y_prob).
    Also note penalty: 0.05 for DNN, 0.1 for Blend, 0 for others
    """
    if model_name == 'xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'lgbm':
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'rf':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'dnn':
        model, auc, ks, y_prob = train_eval_pytorch_dnn(
            X_train.values, y_train.values, X_val.values, y_val.values, device=device
        )
        return model, auc, ks, y_prob

    else:
        raise ValueError(f"Unknown model: {model_name}")


def blend_predictions(probs_list, weights=None):
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    final_prob = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        final_prob += w * p
    return final_prob


# ---------------------------------------------
# 5) Single-step Env with Info dict for AUC, KS, Action
# ---------------------------------------------
class ModelSelectionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, X, y, device='cpu'):
        super().__init__()
        self.device = device

        # Train/val split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.3, random_state=123
        )

        means = X.mean().values
        vars_ = X.var().values
        self.state = np.concatenate([means, vars_])  # observation

        # 5 discrete actions
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state),),
            dtype=np.float32
        )
        self.terminated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.terminated = False
        return self.state.astype(np.float32), {}

    def step(self, action):
        if self.terminated:
            # Already done, no changes
            return self.state.astype(np.float32), 0.0, True, False, {}

        model_names = ['xgb', 'lgbm', 'rf', 'dnn']
        if action < 4:
            chosen_model = model_names[action]
            _, auc_v, ks_v, _ = train_eval_model(
                chosen_model, 
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                device=self.device
            )
            penalty = 0.05 if chosen_model == 'dnn' else 0.0
            reward = (auc_v + ks_v) - penalty
            info = {
                "action_name": chosen_model,
                "AUC": auc_v,
                "KS": ks_v,
                "Penalty": penalty
            }
        else:
            # Blend
            probs_list = []
            model_list = ['xgb', 'lgbm', 'rf', 'dnn']
            for m in model_list:
                _, auc_m, ks_m, prob_m = train_eval_model(
                    m,
                    self.X_train, self.y_train,
                    self.X_val, self.y_val,
                    device=self.device
                )
                probs_list.append(prob_m)
            final_prob = blend_predictions(probs_list)
            auc_v = roc_auc_score(self.y_val, final_prob)
            ks_v = calc_ks_score(self.y_val, final_prob)
            penalty = 0.1
            reward = (auc_v + ks_v) - penalty
            info = {
                "action_name": "blend",
                "AUC": auc_v,
                "KS": ks_v,
                "Penalty": penalty
            }

        self.terminated = True
        return self.state.astype(np.float32), reward, True, False, info


# ---------------------------------------------
# 6) Custom Callback to Track Episode Stats
# ---------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback

class BanditSummaryCallback(BaseCallback):
    """
    A custom callback that records each single-step episode's
    action, AUC, KS, reward. After training, prints a summary table
    like a multi-armed bandit.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.n_actions = 5
        # We'll store episodes in lists keyed by action
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.action_auc = [[] for _ in range(self.n_actions)]
        self.action_ks = [[] for _ in range(self.n_actions)]
        self.action_rewards = [[] for _ in range(self.n_actions)]

    def _on_step(self) -> bool:
        # Because it's a single-step env,
        # we get a "done" every step (episode).
        # We'll parse the info for that step.

        # self.locals["dones"] is an array of bool(s).
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        actions = self.locals["actions"]

        for i, done in enumerate(dones):
            if done:
                # The action for this environment index
                act = actions[i]
                info = infos[i]
                # The reward for this step is in self.locals["rewards"][i]
                rew = self.locals["rewards"][i]

                # Record stats
                self.action_counts[act] += 1
                self.action_rewards[act].append(rew)
                # If the env step returned "AUC" and "KS" in info
                if "AUC" in info and "KS" in info:
                    self.action_auc[act].append(info["AUC"])
                    self.action_ks[act].append(info["KS"])

        return True

    def _on_training_end(self):
        """
        Print final summary after training finishes.
        """
        model_names = ["XGB", "LightGBM", "RandomForest", "DNN", "Blend"]
        # We'll interpret "Q-values" as the average reward for each action
        # across all episodes that used that action:
        Q = np.zeros(self.n_actions, dtype=np.float32)
        for a in range(self.n_actions):
            if len(self.action_rewards[a]) > 0:
                Q[a] = np.mean(self.action_rewards[a])

        best_action = np.argmax(Q)

        print("\n==============================================")
        print("Final Q-values:", Q)
        print(f"Best action index: {best_action}")
        print(f"Best action is: {model_names[best_action]} with estimated Q = {Q[best_action]:.4f}\n")

        print("Detailed AUC/KS/Reward by action:")
        print("--------------------------------------------------")
        for a in range(self.n_actions):
            if len(self.action_rewards[a]) > 0:
                avg_auc = np.mean(self.action_auc[a]) if len(self.action_auc[a]) > 0 else 0.0
                avg_ks = np.mean(self.action_ks[a]) if len(self.action_ks[a]) > 0 else 0.0
                avg_reward = np.mean(self.action_rewards[a])
                count = self.action_counts[a]
                print(f"Action {a} ({model_names[a]}): chosen {count} times")
                print(f"  Mean AUC = {avg_auc:.4f}, Mean KS = {avg_ks:.4f}, Mean Reward = {avg_reward:.4f}\n")
            else:
                print(f"Action {a} ({model_names[a]}): chosen 0 times\n")
        print("==============================================\n")


# ---------------------------------------------
# 7) RL Training & Execution
# ---------------------------------------------
def run_rl_model_selection_pytorch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    # Generate data
    X, y = generate_data(n_samples=10000, n_features=5, random_state=42)

    # Create single-step Gymnasium environment
    env = ModelSelectionEnv(X, y, device=device)

    # Wrap with DummyVecEnv
    def make_env():
        return env
    vec_env = DummyVecEnv([make_env])

    # Create callback
    callback = BanditSummaryCallback()

    # Create DQN
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        tensorboard_log="./rl_tensorboard/"
    )

    # Train with callback
    model.learn(total_timesteps=2000, callback=callback)

    # Evaluate final policy (one step)
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = vec_env.step(action)
    final_reward = rewards[0]
    action_map = ["XGB", "LightGBM", "RandomForest", "DNN", "Blend"]
    print("\n======================================")
    print(f"Final chosen action => {action[0]} ({action_map[action[0]]})")
    print(f"Final step reward => (AUC + KS - penalty) = {final_reward:.4f}")
    print("======================================\n")

if __name__ == "__main__":
    run_rl_model_selection_pytorch()




############read CSV V1#################

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# PyTorch for DNN
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------
# 1) Save and Read Data from CSV
# ---------------------------------------------
"""
# Data generation (can be used if needed)
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

def generate_data(n_samples=10000, n_features=5, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=0.4,
        flip_y=0.2,
        random_state=random_state
    )

    # Introduce mild correlation
    corr = np.array([
        [1.0, 0.2, 0.2, 0.2, 0.2],
        [0.2, 1.0, 0.2, 0.2, 0.2],
        [0.2, 0.2, 1.0, 0.2, 0.2],
        [0.2, 0.2, 0.2, 1.0, 0.2],
        [0.2, 0.2, 0.2, 0.2, 1.0]
    ])
    L = np.linalg.cholesky(corr)
    X = X.dot(L)
    X = StandardScaler().fit_transform(X)

    # Create a DataFrame with features and labels
    df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(n_features)])
    df['label'] = y

    # Save the dataset to CSV
    df.to_csv(r'C:\backupcgi\final_bak\rein_data_binary.csv', index=False)

# Uncomment the next line to generate and save the data if needed
generate_data()
"""

# Read the dataset from CSV
data = pd.read_csv(r'C:\backupcgi\final_bak\rein_data_binary.csv')
X = data.drop('label', axis=1)
y = data['label']

# ---------------------------------------------
# 2) Metrics (AUC, KS)
# ---------------------------------------------
def calc_ks_score(y_true, y_prob):
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).sort_values('y_prob', ascending=False)
    data['cum_pos'] = (data['y_true'] == 1).cumsum()
    data['cum_neg'] = (data['y_true'] == 0).cumsum()
    total_pos = data['y_true'].sum()
    total_neg = (data['y_true'] == 0).sum()
    data['cum_pos_rate'] = data['cum_pos'] / total_pos
    data['cum_neg_rate'] = data['cum_neg'] / total_neg
    data['ks'] = data['cum_pos_rate'] - data['cum_neg_rate']
    return data['ks'].max()

# ---------------------------------------------
# 3) PyTorch DNN Model
# ---------------------------------------------
class DNNModel(nn.Module):
    def __init__(self, input_dim=5):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

def train_eval_pytorch_dnn(X_train, y_train, X_val, y_val,
                           epochs=5, batch_size=64, lr=1e-3, device='cpu'):
    model = DNNModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    dataset_size = len(X_train_t)
    n_batches = (dataset_size // batch_size) + 1

    for epoch in range(epochs):
        indices = torch.randperm(dataset_size)
        X_train_t = X_train_t[indices]
        y_train_t = y_train_t[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            if start_idx >= dataset_size:
                break

            x_batch = X_train_t[start_idx:end_idx]
            y_batch = y_train_t[start_idx:end_idx]

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy().ravel()

    auc = roc_auc_score(y_val, val_preds)
    ks = calc_ks_score(y_val, val_preds)
    return model, auc, ks, val_preds

# ---------------------------------------------
# 4) Helper: Train & Evaluate Various Models
# ---------------------------------------------
def train_eval_model(model_name, X_train, y_train, X_val, y_val, device='cpu'):
    if model_name == 'xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'lgbm':
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'rf':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'dnn':
        model, auc, ks, y_prob = train_eval_pytorch_dnn(
            X_train.values, y_train.values, X_val.values, y_val.values, device=device
        )
        return model, auc, ks, y_prob

    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Continue with the rest of the code unchanged...
# ---------------------------------------------
# 5) Weighted Blending
# ---------------------------------------------
def blend_predictions(probs_list, weights=None):
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    final_prob = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        final_prob += w * p
    return final_prob

def evaluate_action(action, X_train, X_val, y_train, y_val, device='cpu'):
    """
    action: int from 0..4 => (xgb=0, lgbm=1, rf=2, dnn=3, blend=4)
    Returns:
      reward = (auc + ks) - penalty
      auc, ks
    """
    model_names = ['xgb', 'lgbm', 'rf', 'dnn']
    if action < 4:
        chosen_model = model_names[action]
        _, auc_val, ks_val, _ = train_eval_model(chosen_model, X_train, y_train, X_val, y_val, device=device)
        penalty = 0.05 if chosen_model == 'dnn' else 0.0
        reward = (auc_val + ks_val) - penalty
        return reward, auc_val, ks_val
    else:
        # Blend
        probs_list = []
        for m in model_names:
            _, auc_m, ks_m, p = train_eval_model(m, X_train, y_train, X_val, y_val, device=device)
            probs_list.append(p)
        final_prob = blend_predictions(probs_list)
        auc_blend = roc_auc_score(y_val, final_prob)
        ks_blend = calc_ks_score(y_val, final_prob)
        reward = (auc_blend + ks_blend) - 0.1
        return reward, auc_blend, ks_blend

# ---------------------------------------------
# 6) A Simple Multi-Armed Bandit Approach
# ---------------------------------------------
def multi_armed_bandit_model_selection(
    n_episodes=50,
    n_actions=5,
    epsilon=0.06,
    device='cpu'
):
    """
    We have 5 actions (xgb=0, lgbm=1, rf=2, dnn=3, blend=4).
    For each 'episode':
      1) Generate a dataset (X,y) with the chosen seed
      2) Split into train/val
      3) Epsilon-greedy select an action
      4) Evaluate the chosen action => get reward
      5) Update average reward (Q) for that action
    """
    Q = np.zeros(n_actions, dtype=np.float32)
    counts = np.zeros(n_actions, dtype=int)

    # For storing raw AUC, KS, Reward each time an action is chosen
    action_auc_records = [[] for _ in range(n_actions)]
    action_ks_records = [[] for _ in range(n_actions)]
    action_reward_records = [[] for _ in range(n_actions)]

    action_history = []
    reward_history = []

    for episode in range(n_episodes):
        # Generate the data here
        seed = 1000 + episode
        X = data.drop('label', axis=1)  # Features
        y = data['label']  # Labels

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123)

        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q)

        # Evaluate chosen action => get (reward, auc, ks)
        reward, auc_val, ks_val = evaluate_action(
            action, X_train, X_val, y_train, y_val, device=device
        )

        # Update Q (incremental mean)
        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]

        # Store details
        action_history.append(action)
        reward_history.append(reward)
        action_auc_records[action].append(auc_val)
        action_ks_records[action].append(ks_val)
        action_reward_records[action].append(reward)

        print(f"Episode {episode+1}/{n_episodes}, "
              f"Action={action}, Reward={reward:.4f}, Updated Q={Q}")

    return Q, action_history, reward_history, action_auc_records, action_ks_records, action_reward_records

# ---------------------------------------------
# 7) Run the Bandit, then Interpret Results
# ---------------------------------------------
def run_bandit():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device={device}")

    n_episodes = 50
    n_actions = 5
    epsilon = 0.05

    (
        Q,
        actions,
        rewards,
        auc_records,
        ks_records,
        reward_records
    ) = multi_armed_bandit_model_selection(
        n_episodes=n_episodes,
        n_actions=n_actions,
        epsilon=epsilon,
        device=device
    )

    best_action = np.argmax(Q)
    model_names = ["XGB", "LightGBM", "RandomForest", "DNN", "Blend"]

    print("\n========================================")
    print("Interpreting Your Current Results")
    print("========================================\n")

    print("Final Q-values:", Q)
    print(f"Best action index: {best_action}")
    print(f"Best action is: {model_names[best_action]} with estimated Q = {Q[best_action]:.4f}\n")

    print("Detailed AUC/KS/Reward by action:")
    print("--------------------------------------------------")
    for a in range(n_actions):
        if len(auc_records[a]) > 0:
            avg_auc = np.mean(auc_records[a])
            avg_ks = np.mean(ks_records[a])
            avg_reward = np.mean(reward_records[a])
            print(f"Action {a} ({model_names[a]}): chosen {len(auc_records[a])} times")
            print(f"  Mean AUC = {avg_auc:.4f}, Mean KS = {avg_ks:.4f}, Mean Reward = {avg_reward:.4f}\n")
        else:
            print(f"Action {a} ({model_names[a]}): chosen 0 times\n")


if __name__ == "__main__":
    run_bandit()


###########read CSV V2####################
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Gymnasium
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# For callback
from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------
# 1) Read data from CSV file
# ---------------------------------------------
data = pd.read_csv(r'C:\backupcgi\final_bak\rein_data_binary.csv')
X = data.drop('label', axis=1)  # Features
y = data['label']  # Labels

# ---------------------------------------------
# 2) Metrics (AUC, KS)
# ---------------------------------------------
def calc_ks_score(y_true, y_prob):
    data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob}).sort_values('y_prob', ascending=False)
    data['cum_pos'] = (data['y_true'] == 1).cumsum()
    data['cum_neg'] = (data['y_true'] == 0).cumsum()
    total_pos = data['y_true'].sum()
    total_neg = (data['y_true'] == 0).sum()
    data['cum_pos_rate'] = data['cum_pos'] / total_pos
    data['cum_neg_rate'] = data['cum_neg'] / total_neg
    data['ks'] = data['cum_pos_rate'] - data['cum_neg_rate']
    return data['ks'].max()

# ---------------------------------------------
# 3) PyTorch DNN
# ---------------------------------------------
class DNNModel(nn.Module):
    def __init__(self, input_dim=5):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

def train_eval_pytorch_dnn(X_train, y_train, X_val, y_val,
                           epochs=5, batch_size=64, lr=1e-3, device='cpu'):
    model = DNNModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    dataset_size = len(X_train_t)
    n_batches = (dataset_size // batch_size) + 1

    for epoch in range(epochs):
        indices = torch.randperm(dataset_size)
        X_train_t = X_train_t[indices]
        y_train_t = y_train_t[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            if start_idx >= dataset_size:
                break

            x_batch = X_train_t[start_idx:end_idx]
            y_batch = y_train_t[start_idx:end_idx]

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy().ravel()

    auc = roc_auc_score(y_val, val_preds)
    ks = calc_ks_score(y_val, val_preds)
    return model, auc, ks, val_preds

# ---------------------------------------------
# 4) Train & Evaluate Helper
# ---------------------------------------------
def train_eval_model(model_name, X_train, y_train, X_val, y_val, device='cpu'):
    if model_name == 'xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'lgbm':
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'rf':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == 'dnn':
        model, auc, ks, y_prob = train_eval_pytorch_dnn(
            X_train.values, y_train.values, X_val.values, y_val.values, device=device
        )
        return model, auc, ks, y_prob

    else:
        raise ValueError(f"Unknown model name: {model_name}")

def blend_predictions(probs_list, weights=None):
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    final_prob = np.zeros_like(probs_list[0])
    for w, p in zip(weights, probs_list):
        final_prob += w * p
    return final_prob

# ---------------------------------------------
# 5) Single-step Environment
# ---------------------------------------------
class ModelSelectionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, X, y, device='cpu'):
        super().__init__()
        self.device = device

        # Train/val split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.3, random_state=123
        )

        means = X.mean().values
        vars_ = X.var().values
        self.state = np.concatenate([means, vars_])  # observation

        # 5 discrete actions
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state),),
            dtype=np.float32
        )
        self.terminated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.terminated = False
        return self.state.astype(np.float32), {}

    def step(self, action):
        if self.terminated:
            return self.state.astype(np.float32), 0.0, True, False, {}

        model_names = ['xgb', 'lgbm', 'rf', 'dnn']
        if action < 4:
            chosen_model = model_names[action]
            _, auc_v, ks_v, _ = train_eval_model(
                chosen_model, 
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                device=self.device
            )
            penalty = 0.05 if chosen_model == 'dnn' else 0.0
            reward = (auc_v + ks_v) - penalty
            info = {
                "action_name": chosen_model,
                "AUC": auc_v,
                "KS": ks_v,
                "Penalty": penalty
            }
        else:
            # Blend
            probs_list = []
            for m in model_names:
                _, auc_m, ks_m, prob_m = train_eval_model(
                    m,
                    self.X_train, self.y_train,
                    self.X_val, self.y_val,
                    device=self.device
                )
                probs_list.append(prob_m)
            final_prob = blend_predictions(probs_list)
            auc_v = roc_auc_score(self.y_val, final_prob)
            ks_v = calc_ks_score(self.y_val, final_prob)
            penalty = 0.1
            reward = (auc_v + ks_v) - penalty
            info = {
                "action_name": "blend",
                "AUC": auc_v,
                "KS": ks_v,
                "Penalty": penalty
            }

        self.terminated = True
        return self.state.astype(np.float32), reward, True, False, info

# ---------------------------------------------
# 7) RL Training & Execution
# ---------------------------------------------
def run_rl_model_selection_pytorch():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    # Create single-step Gymnasium environment
    env = ModelSelectionEnv(X, y, device=device)

    # Wrap with DummyVecEnv
    def make_env():
        return env
    vec_env = DummyVecEnv([make_env])

    # Create callback
    callback = BanditSummaryCallback()

    # Create DQN
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        tensorboard_log="./rl_tensorboard/"
    )

    # Train with callback
    model.learn(total_timesteps=2000, callback=callback)

    # Evaluate final policy (one step)
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = vec_env.step(action)
    final_reward = rewards[0]
    action_map = ["XGB", "LightGBM", "RandomForest", "DNN", "Blend"]
    print("\n======================================")
    print(f"Final chosen action => {action[0]} ({action_map[action[0]]})")
    print(f"Final step reward => (AUC + KS - penalty) = {final_reward:.4f}")
    print("======================================\n")

if __name__ == "__main__":
    run_rl_model_selection_pytorch()
