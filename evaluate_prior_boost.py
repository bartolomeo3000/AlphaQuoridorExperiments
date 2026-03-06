# ====================
# Prior-Boost Redundancy Evaluator
# ====================
# Answers the question: "Has the NN internalized the BFS/position prior boosts?"
#
# Plays N games:  Model A (boosts ON)  vs  Model B (same weights, boosts OFF)
# Both load the same .pt file (default: best.pt) so the only difference is the
# prior modification applied at inference time.
#
# If Model B (no boosts) scores >= THRESHOLD, the NN has learned the bias on its own
# and you can safely set POSITION_PRIOR_BOOST = BFS_MOVE_BOOST = 1.0 permanently.
#
# Usage:
#   python evaluate_prior_boost.py                  # uses best.pt, 40 games
#   python evaluate_prior_boost.py --model latest   # uses latest.pt
#   python evaluate_prior_boost.py --games 80       # more games for cleaner signal

import argparse
import io
import os
import sys
from copy import deepcopy

import numpy as np
import torch
import multiprocessing as mp

from game import State, _get_blocked_edges, _bfs_goal_distances
from dual_network import DualNetwork, DN_INPUT_SHAPE, DN_OUTPUT_SIZE, load_model
from config import (
    MODEL_DIR, EN_TEMPERATURE, EN_TEMP_CUTOFF,
    POSITION_PRIOR_BOOST, BFS_MOVE_BOOST,
    PV_EVALUATE_COUNT, DIRICHLET_ALPHA,
)
from pv_mcts import pv_mcts_scores, nodes_to_scores
from math import sqrt


# ── Custom predict with configurable boosts ───────────────────────────────────

def predict_with_boosts(model, state, pos_boost, bfs_boost):
    a, b, c = DN_INPUT_SHAPE
    x = np.array(state.pieces_array(), dtype=np.float32).reshape(c, a, b)
    x = torch.from_numpy(x).unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)
    model.eval()
    with torch.no_grad():
        p, v = model(x)

    legal = list(state.legal_actions())
    policies = p[0].cpu().numpy()[legal]
    N = state.N

    if pos_boost != 1.0:
        for i, action in enumerate(legal):
            if action < N * N:
                policies[i] *= pos_boost

    if bfs_boost != 1.0:
        walls_t = tuple(state.walls)
        h_e, v_e = _get_blocked_edges(N, walls_t)
        dist = _bfs_goal_distances(N, h_e, v_e, 0)
        current_dist = dist[state.player[0]]
        for i, action in enumerate(legal):
            if action < N * N and dist[action] < current_dist:
                policies[i] *= bfs_boost

    total = np.sum(policies)
    policies /= total if total else 1
    value = v[0][0].cpu().item()
    return policies, value


# ── MCTS using a specific boost config ───────────────────────────────────────

def mcts_scores_boosted(model, state, temperature, pos_boost, bfs_boost):
    """Minimal MCTS reusing Node structure but calling our configurable predict."""
    C_PUCT = 1.0

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
                self.w += value
                self.n += 1
                return value
            if not self.child_nodes:
                policies, value = predict_with_boosts(model, self.state, pos_boost, bfs_boost)
                self.w += value
                self.n += 1
                self.child_nodes = [
                    Node(self.state.next(a), pol)
                    for a, pol in zip(self.state.legal_actions(), policies)
                ]
                return value
            value = -self.next_child().evaluate()
            self.w += value
            self.n += 1
            return value

        def next_child(self):
            t = sum(c.n for c in self.child_nodes)
            scores = [
                (-c.w / c.n if c.n else 0.0) + C_PUCT * c.p * sqrt(t) / (1 + c.n)
                for c in self.child_nodes
            ]
            return self.child_nodes[np.argmax(scores)]

    root = Node(state, 0)
    root.evaluate()
    for _ in range(PV_EVALUATE_COUNT - 1):
        root.evaluate()

    counts = [c.n for c in root.child_nodes]
    if temperature == 0:
        idx = np.argmax(counts)
        out = np.zeros(len(counts))
        out[idx] = 1.0
        return out
    counts = np.array(counts, dtype=np.float64)
    counts = counts ** (1.0 / temperature)
    return counts / counts.sum()


# ── Worker (must be top-level for multiprocessing spawn on Windows) ───────────

def _worker(args):
    sd_bytes, game_idx, pos_boost_a, bfs_boost_a, pos_boost_b, bfs_boost_b, temperature, temp_cutoff = args

    model_a = DualNetwork()
    model_a.load_state_dict(torch.load(io.BytesIO(sd_bytes), map_location='cpu'))
    model_a.eval()
    model_b = DualNetwork()
    model_b.load_state_dict(torch.load(io.BytesIO(sd_bytes), map_location='cpu'))
    model_b.eval()

    # Even games: A plays first; odd games: B plays first (eliminates first-move advantage)
    a_is_first = (game_idx % 2 == 0)

    state = State()
    move_count = 0
    while not state.is_done():
        t = temperature if move_count < temp_cutoff else 0.0
        if state.is_first_player() == a_is_first:
            scores = mcts_scores_boosted(model_a, deepcopy(state), t, pos_boost_a, bfs_boost_a)
        else:
            scores = mcts_scores_boosted(model_b, deepcopy(state), t, pos_boost_b, bfs_boost_b)
        action = np.random.choice(state.legal_actions(), p=scores)
        state = state.next(action)
        move_count += 1

    # Point for Model A
    if state.is_draw():
        point_a = 0.5
    else:
        first_player_won = not state.is_first_player()  # last mover loses
        a_won = (first_player_won == a_is_first)
        point_a = 1.0 if a_won else 0.0

    return point_a, move_count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate whether prior boosts are still needed.')
    parser.add_argument('--model',     default='best',   help='Model to load: "best" or "latest" (default: best)')
    parser.add_argument('--games',     type=int, default=40, help='Number of evaluation games (default: 40)')
    parser.add_argument('--threshold', type=float, default=0.45, help='Score for no-boost side to be considered independent (default: 0.45)')
    args = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, f'{args.model}.pt')
    if not os.path.exists(model_path):
        sys.exit(f'Model not found: {model_path}')

    print(f'Model:        {model_path}')
    print(f'Games:        {args.games}')
    print(f'Boosts ON:    POSITION_PRIOR_BOOST={POSITION_PRIOR_BOOST}  BFS_MOVE_BOOST={BFS_MOVE_BOOST}')
    print(f'Boosts OFF:   POSITION_PRIOR_BOOST=1.0  BFS_MOVE_BOOST=1.0')
    print(f'Threshold:    no-boost score >= {args.threshold} → boosts no longer needed')
    print()

    # Serialize weights once
    buf = io.BytesIO()
    torch.save(load_model(model_path).state_dict(), buf)
    sd_bytes = buf.getvalue()

    worker_args = [
        (
            sd_bytes, i,
            POSITION_PRIOR_BOOST, BFS_MOVE_BOOST,   # Model A: boosts ON
            1.0, 1.0,                                 # Model B: boosts OFF
            EN_TEMPERATURE, EN_TEMP_CUTOFF,
        )
        for i in range(args.games)
    ]

    pool = mp.Pool()
    completed = wins_a = draws = wins_b = 0
    total_plies = 0

    for point_a, plies in pool.imap_unordered(_worker, worker_args):
        completed += 1
        total_plies += plies
        if point_a > 0.6:
            wins_a += 1
        elif point_a < 0.4:
            wins_b += 1
        else:
            draws += 1
        score_b = (wins_b + 0.5 * draws) / completed
        print(f'\r{completed}/{args.games}  '
              f'Boosted wins: {wins_a}  Draws: {draws}  No-boost wins: {wins_b}  '
              f'| No-boost score: {score_b:.3f}', end='')

    pool.close()
    pool.join()
    print()

    score_a = (wins_a + 0.5 * draws) / args.games
    score_b = (wins_b + 0.5 * draws) / args.games
    avg_plies = total_plies / args.games

    print(f'\n── Results ──────────────────────────────────────────────')
    print(f'  Boosted    score: {score_a:.4f}  ({wins_a}W / {draws}D / {wins_b}L)')
    print(f'  No-boost   score: {score_b:.4f}  ({wins_b}W / {draws}D / {wins_a}L)')
    print(f'  Avg game length:  {avg_plies:.1f} plies')
    print()
    if score_b >= args.threshold:
        print(f'✓ No-boost score {score_b:.3f} >= {args.threshold}  →  NN has internalized the biases.')
        print(f'  Safe to set POSITION_PRIOR_BOOST = BFS_MOVE_BOOST = 1.0 in config.py')
    else:
        print(f'✗ No-boost score {score_b:.3f} < {args.threshold}  →  Boosts still helping.')
        print(f'  Keep current values and re-run after more training cycles.')


if __name__ == '__main__':
    main()
