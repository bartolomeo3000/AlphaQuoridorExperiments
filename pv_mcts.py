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
    PV_EVALUATE_COUNT, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, POSITION_PRIOR_BOOST,
    BFS_MOVE_BOOST, BFS_MOVE_PENALTY, BFS_ADVANCE_FLOOR, BFS_RETREAT_CEILING,
    FPU_REDUCTION, BFS_PUCT_RETREAT_PENALTY, BFS_PUCT_ADVANCE_BONUS, BFS_WALL_PUCT_SCALE,
)

# Inference
def predict(model, state,
            pos_boost=None, bfs_boost=None, bfs_penalty=None,
            bfs_floor=None, bfs_retreat_ceiling=None):
    # Resolve overrides; None → fall back to global config
    _pos_boost  = POSITION_PRIOR_BOOST if pos_boost          is None else pos_boost
    _bfs_boost  = BFS_MOVE_BOOST       if bfs_boost          is None else bfs_boost
    _bfs_pen    = BFS_MOVE_PENALTY     if bfs_penalty        is None else bfs_penalty
    _bfs_floor  = BFS_ADVANCE_FLOOR    if bfs_floor          is None else bfs_floor
    _bfs_ceil   = BFS_RETREAT_CEILING  if bfs_retreat_ceiling is None else bfs_retreat_ceiling

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
    if _pos_boost != 1.0:
        for i, action in enumerate(legal):
            if action < N * N:
                policies[i] *= _pos_boost

    # Extra boost for the pawn move(s) that advance along the BFS-shortest path to goal
    # and penalty for pawn moves that retreat (increase BFS distance).
    if _bfs_boost != 1.0 or _bfs_pen != 1.0:
        walls_t = tuple(state.walls)
        h_e, v_e = _get_blocked_edges(N, walls_t)
        dist = _bfs_goal_distances(N, h_e, v_e, 0)     # dist to row 0 = current player's goal
        current_dist = dist[state.player[0]]
        for i, action in enumerate(legal):
            if action < N * N:
                if dist[action] < current_dist:
                    policies[i] *= _bfs_boost
                elif dist[action] > current_dist:
                    policies[i] *= _bfs_pen

    policies /= np.sum(policies) if np.sum(policies) else 1  # normalise to sum to 1

    # Floor: guarantee each advancing pawn move a minimum probability share,
    # then renormalise again. Fixes cases where the NN assigns near-zero prior
    # to an advancing move — multiplication alone cannot rescue a true zero.
    if _bfs_floor > 0.0:
        walls_t = tuple(state.walls)  # may already be computed above; compute again if not
        if not (_bfs_boost != 1.0 or _bfs_pen != 1.0):
            h_e, v_e = _get_blocked_edges(N, walls_t)
            dist = _bfs_goal_distances(N, h_e, v_e, 0)
            current_dist = dist[state.player[0]]
        for i, action in enumerate(legal):
            if action < N * N and dist[action] < current_dist:
                if policies[i] < _bfs_floor:
                    policies[i] = _bfs_floor
        policies /= np.sum(policies)

    # Ceiling: cap each retreating pawn move to prevent the value head from
    # rescuing a move the policy correctly penalised.
    if _bfs_ceil < 1.0:
        if not (_bfs_boost != 1.0 or _bfs_pen != 1.0 or _bfs_floor > 0.0):
            walls_t = tuple(state.walls)
            h_e, v_e = _get_blocked_edges(N, walls_t)
            dist = _bfs_goal_distances(N, h_e, v_e, 0)
            current_dist = dist[state.player[0]]
        capped = False
        for i, action in enumerate(legal):
            if action < N * N and dist[action] > current_dist:
                if policies[i] > _bfs_ceil:
                    policies[i] = _bfs_ceil
                    capped = True
        if capped:
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
def pv_mcts_scores(model, state, temperature, add_noise=False, use_q_selection=False, return_root_q=False,
                   sims=None, pos_boost=None, bfs_boost=None, bfs_penalty=None,
                   bfs_floor=None, bfs_retreat_ceiling=None,
                   bfs_retreat_penalty=None, bfs_advance_bonus=None, bfs_wall_scale=None):
    # Resolve per-call overrides; None → global config
    _sims    = PV_EVALUATE_COUNT        if sims                is None else sims
    _retreat = BFS_PUCT_RETREAT_PENALTY if bfs_retreat_penalty is None else bfs_retreat_penalty
    _advance = BFS_PUCT_ADVANCE_BONUS   if bfs_advance_bonus   is None else bfs_advance_bonus
    _wall    = BFS_WALL_PUCT_SCALE      if bfs_wall_scale      is None else bfs_wall_scale
    # Define Monte Carlo Tree Search node
    class Node:
        # Initialize node
        def __init__(self, state, p, bfs_puct_adj=0.0):
            self.state = state # State
            self.p = p # Policy
            self.w = 0 # Cumulative value
            self.n = 0 # Number of simulations
            self.child_nodes = None  # Child nodes
            self.bfs_puct_adj = bfs_puct_adj  # Pre-computed signed PUCT offset (added directly)

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
                policies, value = predict(model, self.state,
                    pos_boost=pos_boost, bfs_boost=bfs_boost,
                    bfs_penalty=bfs_penalty, bfs_floor=bfs_floor,
                    bfs_retreat_ceiling=bfs_retreat_ceiling)

                # Update cumulative value and number of simulations
                self.w += value
                self.n += 1

                # Compute BFS-based PUCT adjustments once at expansion
                N = self.state.N
                need_bfs = (_retreat != 0.0 or _advance != 0.0 or _wall != 0.0)
                if need_bfs:
                    walls_t = tuple(self.state.walls)
                    h_e, v_e = _get_blocked_edges(N, walls_t)
                    dist = _bfs_goal_distances(N, h_e, v_e, 0)
                    cur = dist[self.state.player[0]]
                    if _wall != 0.0:
                        # Enemy BFS: rotated walls + goal row 0 — mirrors bfs_distances() / next()
                        walls_rot = tuple(reversed(walls_t))
                        h_r, v_r = _get_blocked_edges(N, walls_rot)
                        dist_opp = _bfs_goal_distances(N, h_r, v_r, 0)
                        opp_cur = dist_opp[self.state.enemy[0]]
                    else:
                        dist_opp = None; opp_cur = 0
                else:
                    dist = None; cur = 0; dist_opp = None; opp_cur = 0

                # Expand child nodes with pre-computed PUCT adjustments
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    child_state = self.state.next(action)
                    if dist is not None and action < N * N:
                        delta = dist[action] - cur
                        if delta < 0:   # advancing toward goal
                            adj = _advance * (-delta)
                        elif delta > 0: # retreating from goal
                            adj = -_retreat * delta
                        else:
                            adj = 0.0
                    elif dist_opp is not None and action >= N * N:
                        # Wall move: net bonus = scale × (opp_increase - our_increase)²
                        h_e_w, v_e_w = _get_blocked_edges(N, tuple(child_state.walls))
                        opp_new = _bfs_goal_distances(N, h_e_w, v_e_w, 0)[child_state.player[0]]
                        opp_delta = opp_new - opp_cur
                        new_walls = list(walls_t)
                        if action < N * N + (N - 1) ** 2:
                            new_walls[action - N * N] = 1
                        else:
                            new_walls[action - N * N - (N - 1) ** 2] = 2
                        h_e_w2, v_e_w2 = _get_blocked_edges(N, tuple(new_walls))
                        our_new = _bfs_goal_distances(N, h_e_w2, v_e_w2, 0)[self.state.player[0]]
                        net_delta = opp_delta - (our_new - cur)
                        adj = _wall * net_delta if net_delta > 0 else 0.0
                    else:
                        adj = 0.0
                    self.child_nodes.append(Node(child_state, policy, adj))
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
            fpu = (self.w / self.n - FPU_REDUCTION) if (FPU_REDUCTION is not None and self.n) else 0.0
            pucb_values = []
            for child_node in self.child_nodes:
                q = (-child_node.w / child_node.n if child_node.n else fpu)
                pucb_values.append(
                    q
                    + C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n)
                    + child_node.bfs_puct_adj
                )

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
    for _ in range(_sims - 1):
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
    if return_root_q:
        root_q = root_node.w / root_node.n if root_node.n else 0.0
        return scores, root_q
    return scores

def pv_mcts_full(model, state, rollouts):
    """Run MCTS and return (visit_probs_over_legal_actions, root_q_value).

    Unlike pv_mcts_scores, always returns raw visit proportions (never
    collapses to one-hot) and also returns the root Q value (tanh scale).
    """
    class Node:
        def __init__(self, state, p, bfs_puct_adj=0.0):
            self.state = state
            self.p = p
            self.w = 0.0
            self.n = 0
            self.child_nodes = None
            self.bfs_puct_adj = bfs_puct_adj

        def evaluate(self):
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0
                self.w += value; self.n += 1
                return value
            if not self.child_nodes:
                policies, value = predict(model, self.state)
                self.w += value; self.n += 1
                N = self.state.N
                need_bfs = (BFS_PUCT_RETREAT_PENALTY != 0.0 or BFS_PUCT_ADVANCE_BONUS != 0.0 or BFS_WALL_PUCT_SCALE != 0.0)
                if need_bfs:
                    walls_t = tuple(self.state.walls)
                    h_e, v_e = _get_blocked_edges(N, walls_t)
                    dist = _bfs_goal_distances(N, h_e, v_e, 0)
                    cur = dist[self.state.player[0]]
                    if BFS_WALL_PUCT_SCALE != 0.0:
                        # Enemy BFS: rotated walls + goal row 0 — mirrors bfs_distances() / next()
                        walls_rot = tuple(reversed(walls_t))
                        h_r, v_r = _get_blocked_edges(N, walls_rot)
                        dist_opp = _bfs_goal_distances(N, h_r, v_r, 0)
                        opp_cur = dist_opp[self.state.enemy[0]]
                    else:
                        dist_opp = None; opp_cur = 0
                else:
                    dist = None; cur = 0; dist_opp = None; opp_cur = 0
                child_nodes_tmp = []
                for a, p in zip(self.state.legal_actions(), policies):
                    cs = self.state.next(a)
                    if dist is not None and a < N * N:
                        d = dist[a] - cur
                        adj = BFS_PUCT_ADVANCE_BONUS * (-d) if d < 0 else (-BFS_PUCT_RETREAT_PENALTY * d if d > 0 else 0.0)
                    elif dist_opp is not None and a >= N * N:
                        # Wall move: net bonus = scale × (opp_increase - our_increase)²
                        h_e_w, v_e_w = _get_blocked_edges(N, tuple(cs.walls))
                        opp_new = _bfs_goal_distances(N, h_e_w, v_e_w, 0)[cs.player[0]]
                        opp_delta = opp_new - opp_cur
                        new_walls = list(walls_t)
                        if a < N * N + (N - 1) ** 2:
                            new_walls[a - N * N] = 1
                        else:
                            new_walls[a - N * N - (N - 1) ** 2] = 2
                        h_e_w2, v_e_w2 = _get_blocked_edges(N, tuple(new_walls))
                        our_new = _bfs_goal_distances(N, h_e_w2, v_e_w2, 0)[self.state.player[0]]
                        net_delta = opp_delta - (our_new - cur)
                        adj = BFS_WALL_PUCT_SCALE * net_delta if net_delta > 0 else 0.0
                    else:
                        adj = 0.0
                    child_nodes_tmp.append(Node(cs, p, adj))
                self.child_nodes = child_nodes_tmp
                return value
            value = -self.next_child_node().evaluate()
            self.w += value; self.n += 1
            return value

        def next_child_node(self):
            t = sum(c.n for c in self.child_nodes)
            fpu = (self.w / self.n - FPU_REDUCTION) if (FPU_REDUCTION is not None and self.n) else 0.0
            return self.child_nodes[np.argmax([
                (-c.w / c.n if c.n else fpu)
                + C_PUCT * c.p * sqrt(t) / (1 + c.n)
                + c.bfs_puct_adj
                for c in self.child_nodes
            ])]

    root = Node(state, 0)
    for _ in range(rollouts):
        root.evaluate()

    visits = np.array([c.n for c in root.child_nodes], dtype=np.float32)
    total = visits.sum()
    visit_probs = (visits / total).tolist() if total > 0 else (np.ones(len(visits)) / len(visits)).tolist()
    root_q = root.w / root.n if root.n else 0.0
    # Per-child Q from parent's view: positive = good for current player
    action_q = [float(-c.w / c.n) if c.n else None for c in root.child_nodes]
    return visit_probs, float(root_q), action_q


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
