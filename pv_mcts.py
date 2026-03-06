# ====================
# Monte Carlo Tree Search Implementation
# ====================

# Import packages
from game import State, _get_blocked_edges, _bfs_goal_distances
from dual_network import DN_INPUT_SHAPE, load_model
from math import sqrt
from pathlib import Path
import numpy as np
import torch
from copy import deepcopy
import random

from config import (
    PV_EVALUATE_COUNT, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, POSITION_PRIOR_BOOST, BFS_MOVE_BOOST, BFS_MOVE_PENALTY, BFS_ADVANCE_FLOOR,
)

# Inference
def predict(model, state):
    # Reshape input data for inference — PyTorch uses NCHW (N, C, H, W)
    a, b, c = DN_INPUT_SHAPE  # a=H=3, b=W=3, c=C=6
    x = np.array(state.pieces_array(), dtype=np.float32).reshape(c, a, b)  # (C, H, W)
    x = torch.from_numpy(x).unsqueeze(0)  # (1, C, H, W)
    device = next(model.parameters()).device
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        p, v = model(x)

    # Get policy — only legal moves
    legal = list(state.legal_actions())
    policies = p[0].cpu().numpy()[legal]

    # Boost prior probability of position moves to compensate for wall actions
    # dominating the action space numerically
    N = state.N
    if POSITION_PRIOR_BOOST != 1.0:
        for i, action in enumerate(legal):
            if action < N * N:
                policies[i] *= POSITION_PRIOR_BOOST

    # Extra boost for the pawn move(s) that advance along the BFS-shortest path to goal
    # and penalty for pawn moves that retreat (increase BFS distance).
    if BFS_MOVE_BOOST != 1.0 or BFS_MOVE_PENALTY != 1.0:
        walls_t = tuple(state.walls)
        h_e, v_e = _get_blocked_edges(N, walls_t)
        dist = _bfs_goal_distances(N, h_e, v_e, 0)     # dist to row 0 = current player's goal
        current_dist = dist[state.player[0]]
        for i, action in enumerate(legal):
            if action < N * N:
                if dist[action] < current_dist:
                    policies[i] *= BFS_MOVE_BOOST
                elif dist[action] > current_dist:
                    policies[i] *= BFS_MOVE_PENALTY

    policies /= np.sum(policies) if np.sum(policies) else 1  # normalise to sum to 1

    # Floor: guarantee each advancing pawn move a minimum probability share,
    # then renormalise again. Fixes cases where the NN assigns near-zero prior
    # to an advancing move — multiplication alone cannot rescue a true zero.
    if BFS_ADVANCE_FLOOR > 0.0:
        walls_t = tuple(state.walls)  # may already be computed above; compute again if not
        if not (BFS_MOVE_BOOST != 1.0 or BFS_MOVE_PENALTY != 1.0):
            h_e, v_e = _get_blocked_edges(N, walls_t)
            dist = _bfs_goal_distances(N, h_e, v_e, 0)
            current_dist = dist[state.player[0]]
        for i, action in enumerate(legal):
            if action < N * N and dist[action] < current_dist:
                if policies[i] < BFS_ADVANCE_FLOOR:
                    policies[i] = BFS_ADVANCE_FLOOR
        policies /= np.sum(policies)

    # Get value
    value = v[0][0].cpu().item()
    return policies, value

# Convert list of nodes to list of scores
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

# Get Monte Carlo Tree Search scores
def pv_mcts_scores(model, state, temperature, add_noise=False, use_q_selection=False):
    # Define Monte Carlo Tree Search node
    class Node:
        # Initialize node
        def __init__(self, state, p):
            self.state = state # State
            self.p = p # Policy
            self.w = 0 # Cumulative value
            self.n = 0 # Number of simulations
            self.child_nodes = None  # Child nodes

        # Calculate value of the state
        def evaluate(self):
            # If the game is over
            if self.state.is_done():
                # Get value from the game result
                value = -1 if self.state.is_lose() else 0

                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1
                return value

            # If there are no child nodes
            if not self.child_nodes:
                # Get policy and value from neural network inference
                policies, value = predict(model, self.state)

                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1

                # Expand child nodes
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(Node(self.state.next(action), policy))
                return value

            # If there are child nodes
            else:
                # Get value from the evaluation of the child node with the maximum arc evaluation value
                value = -self.next_child_node().evaluate()

                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1
                return value

        # Get child node with the maximum arc evaluation value
        def next_child_node(self):
            # Calculate arc evaluation value
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                    C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

            # Return child node with the maximum arc evaluation value
            return self.child_nodes[np.argmax(pucb_values)]

    # Create a node for the current state
    root_node = Node(state, 0)

    # Force root expansion so we can inject noise into child priors before searching
    root_node.evaluate()

    # Dirichlet noise: encourages exploration at the root during self-play
    if add_noise and root_node.child_nodes:
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root_node.child_nodes))
        for child, n in zip(root_node.child_nodes, noise):
            child.p = (1 - DIRICHLET_EPSILON) * child.p + DIRICHLET_EPSILON * n

    # Perform remaining evaluations
    for _ in range(PV_EVALUATE_COUNT - 1):
        root_node.evaluate()

    # Probability distribution of legal moves
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        if use_q_selection:
            # Select by highest Q value (mean backed-up value, no exploration bonus).
            # Unvisited children (n=0) get -inf so they are never chosen over visited ones.
            q_values = [(-c.w / c.n if c.n else -float('inf')) for c in root_node.child_nodes]
            action = np.argmax(q_values)
        else:
            # Default: select most-visited child (standard AlphaZero)
            action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else: # Add variation with Boltzmann distribution over visit counts
        scores = boltzman(scores, temperature)
    return scores

def pv_mcts_full(model, state, rollouts):
    """Run MCTS and return (visit_probs_over_legal_actions, root_q_value).

    Unlike pv_mcts_scores, always returns raw visit proportions (never
    collapses to one-hot) and also returns the root Q value (tanh scale).
    """
    class Node:
        def __init__(self, state, p):
            self.state = state
            self.p = p
            self.w = 0.0
            self.n = 0
            self.child_nodes = None

        def evaluate(self):
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0
                self.w += value; self.n += 1
                return value
            if not self.child_nodes:
                policies, value = predict(model, self.state)
                self.w += value; self.n += 1
                self.child_nodes = [
                    Node(self.state.next(a), p)
                    for a, p in zip(self.state.legal_actions(), policies)
                ]
                return value
            value = -self.next_child_node().evaluate()
            self.w += value; self.n += 1
            return value

        def next_child_node(self):
            t = sum(c.n for c in self.child_nodes)
            return self.child_nodes[np.argmax([
                (-c.w / c.n if c.n else 0.0) + C_PUCT * c.p * sqrt(t) / (1 + c.n)
                for c in self.child_nodes
            ])]

    root = Node(state, 0)
    for _ in range(rollouts):
        root.evaluate()

    visits = np.array([c.n for c in root.child_nodes], dtype=np.float32)
    total = visits.sum()
    visit_probs = (visits / total).tolist() if total > 0 else (np.ones(len(visits)) / len(visits)).tolist()
    root_q = root.w / root.n if root.n else 0.0
    return visit_probs, float(root_q)


# Action selection with Monte Carlo Tree Search
def pv_mcts_action(model, temperature=0, add_noise=False, temp_cutoff=None, use_q_selection=False):
    move_count = [0]  # mutable counter shared in closure
    def pv_mcts_action(state):
        if temp_cutoff is not None and move_count[0] >= temp_cutoff:
            t = 0
        else:
            t = temperature
        move_count[0] += 1
        scores = pv_mcts_scores(model, deepcopy(state), t, add_noise=add_noise,
                                use_q_selection=use_q_selection)

        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

# Boltzmann distribution
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]

def random_action():
    def random_action(state):
        legal_actions = state.legal_actions()
        action = random.randint(0, len(legal_actions) - 1)

        return legal_actions[action]
    return random_action

# Confirm operation
if __name__ == '__main__':
    # Load model
    path = sorted(Path('./model').glob('*.pt'))[-1]
    model = load_model(str(path))

    # Generate state
    state = State()

    # Create function to get actions with Monte Carlo Tree Search
    next_action = pv_mcts_action(model, 1.0)

    # Loop until the game is over
    while True:
        # If the game is over
        if state.is_done():
            break

        # Get action
        action = next_action(state)

        # Get next state
        state = state.next(action)

        # Print state
        print(state)
