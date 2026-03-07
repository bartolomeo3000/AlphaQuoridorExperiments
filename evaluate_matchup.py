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
    MODEL_DIR, EN_TEMPERATURE, EN_TEMP_CUTOFF, EN_FORCED_OPENING,
    POSITION_PRIOR_BOOST, BFS_MOVE_BOOST,
    PV_EVALUATE_COUNT, C_PUCT, DIRICHLET_ALPHA,
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

def mcts_scores_boosted(model, state, temperature, pos_boost, bfs_boost, sims):
    """Minimal MCTS reusing Node structure but calling our configurable predict."""

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
    for _ in range(sims - 1):
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
    sd_bytes_a, sd_bytes_b, game_idx, pos_boost_a, bfs_boost_a, pos_boost_b, bfs_boost_b, sims_a, sims_b, temperature, temp_cutoff, opening_actions = args

    model_a = DualNetwork()
    model_a.load_state_dict(torch.load(io.BytesIO(sd_bytes_a), map_location='cpu'))
    model_a.eval()
    model_b = DualNetwork()
    model_b.load_state_dict(torch.load(io.BytesIO(sd_bytes_b), map_location='cpu'))
    model_b.eval()

    # Even games: A plays first; odd games: B plays first (eliminates first-move advantage)
    a_is_first = (game_idx % 2 == 0)

    state = State()
    move_count = 0
    all_actions = []

    # Replay the pre-generated opening (same for both games in a pair;
    # even game_idx = A first, odd = B first — bias cancels within pair).
    for action in opening_actions:
        if state.is_done():
            break
        state = state.next(action)
        all_actions.append(int(action))
        move_count += 1

    while not state.is_done():
        t = temperature if move_count < temp_cutoff else 0.0
        if state.is_first_player() == a_is_first:
            scores = mcts_scores_boosted(model_a, deepcopy(state), t, pos_boost_a, bfs_boost_a, sims_a)
        else:
            scores = mcts_scores_boosted(model_b, deepcopy(state), t, pos_boost_b, bfs_boost_b, sims_b)
        action = int(np.random.choice(state.legal_actions(), p=scores))
        all_actions.append(action)
        state = state.next(action)
        move_count += 1

    # Point for Model A
    if state.is_draw():
        point_a = 0.5
    else:
        first_player_won = not state.is_first_player()  # last mover loses
        a_won = (first_player_won == a_is_first)
        point_a = 1.0 if a_won else 0.0

    return point_a, move_count, all_actions, a_is_first


# ── Callable API for the web dashboard ──────────────────────────────────────

def run_matchup(cfg, on_game=None, cancel_flag=None):
    """
    Run a matchup and stream results via on_game callback.

    cfg keys:
        model_a, model_b  — absolute paths to .pt files
        games             — total games (must be even for pairing)
        pos_a, bfs_a, sims_a
        pos_b, bfs_b, sims_b

    on_game(game_dict) is called from the pool thread for each completed game.
    cancel_flag is a threading.Event; if set, the pool is terminated early.
    Returns a summary dict.
    """
    def _to_bytes(path):
        m = load_model(path)
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        del m
        return buf.getvalue()

    sd_bytes_a = _to_bytes(cfg['model_a'])
    sd_bytes_b = _to_bytes(cfg['model_b'])

    pos_a  = cfg.get('pos_a',  POSITION_PRIOR_BOOST)
    bfs_a  = cfg.get('bfs_a',  BFS_MOVE_BOOST)
    sims_a = cfg.get('sims_a', PV_EVALUATE_COUNT)
    pos_b  = cfg.get('pos_b',  POSITION_PRIOR_BOOST)
    bfs_b  = cfg.get('bfs_b',  BFS_MOVE_BOOST)
    sims_b = cfg.get('sims_b', PV_EVALUATE_COUNT)

    n_pairs = cfg['games'] // 2
    worker_args = []
    for pair_idx in range(n_pairs):
        state = State()
        opening_actions = []
        for _ in range(EN_FORCED_OPENING):
            if state.is_done():
                break
            action = int(np.random.choice(state.legal_actions()))
            opening_actions.append(action)
            state = state.next(action)
        for _, game_idx in enumerate([pair_idx * 2, pair_idx * 2 + 1]):
            worker_args.append((
                sd_bytes_a, sd_bytes_b, game_idx,
                pos_a, bfs_a, pos_b, bfs_b,
                sims_a, sims_b,
                EN_TEMPERATURE, EN_TEMP_CUTOFF,
                opening_actions,
            ))

    pool = mp.Pool()
    wins_a = draws = wins_b = 0
    cancelled = False

    for point_a, plies, all_actions, a_is_first in pool.imap_unordered(_worker, worker_args):
        if point_a > 0.6:
            wins_a += 1
            result = 'a_win'
        elif point_a < 0.4:
            wins_b += 1
            result = 'b_win'
        else:
            draws += 1
            result = 'draw'

        if on_game:
            on_game({
                'actions':  all_actions,
                'a_first':  a_is_first,
                'result':   result,
                'plies':    plies,
            })

        if cancel_flag and cancel_flag.is_set():
            pool.terminate()
            cancelled = True
            break

    if not cancelled:
        pool.close()
    pool.join()

    completed = wins_a + draws + wins_b
    score_a = (wins_a + 0.5 * draws) / completed if completed else 0.0
    score_b = (wins_b + 0.5 * draws) / completed if completed else 0.0
    return {
        'wins_a': wins_a, 'draws': draws, 'wins_b': wins_b,
        'completed': completed, 'total': cfg['games'],
        'score_a': round(score_a, 4),
        'score_b': round(score_b, 4),
        'cancelled': cancelled,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate whether prior boosts are still needed.')
    parser.add_argument('--model',   default='best', help='Model A to load: name without .pt, or full path (default: best)')
    parser.add_argument('--model-b', default=None,   help='Model B to load: name without .pt, or full path (default: same as --model)')
    parser.add_argument('--games',     type=int, default=40, help='Number of evaluation games (default: 40)')
    parser.add_argument('--threshold', type=float, default=0.45, help='Score for Model B to be considered better (default: 0.45)')
    parser.add_argument('--pos-a',  type=float, default=POSITION_PRIOR_BOOST, help='POSITION_PRIOR_BOOST for Model A (default: config value)')
    parser.add_argument('--bfs-a',  type=float, default=BFS_MOVE_BOOST,       help='BFS_MOVE_BOOST for Model A (default: config value)')
    parser.add_argument('--pos-b',  type=float, default=POSITION_PRIOR_BOOST,  help='POSITION_PRIOR_BOOST for Model B (default: config value)')
    parser.add_argument('--bfs-b',  type=float, default=BFS_MOVE_BOOST,        help='BFS_MOVE_BOOST for Model B (default: config value)')
    parser.add_argument('--sims-a', type=int,   default=PV_EVALUATE_COUNT,    help=f'MCTS simulations for Model A (default: {PV_EVALUATE_COUNT})')
    parser.add_argument('--sims-b', type=int,   default=PV_EVALUATE_COUNT,    help=f'MCTS simulations for Model B (default: {PV_EVALUATE_COUNT})')
    args = parser.parse_args()

    pos_boost_a, bfs_boost_a = args.pos_a, args.bfs_a
    pos_boost_b, bfs_boost_b = args.pos_b, args.bfs_b
    sims_a, sims_b = args.sims_a, args.sims_b

    def resolve(name):
        """Accept a bare name like 'best' or 'cycle_0020', or a full path."""
        if os.path.isabs(name) or os.sep in name or '/' in name:
            return name
        return os.path.join(MODEL_DIR, f'{name}.pt')

    model_path_a = resolve(args.model)
    model_path_b = resolve(args.model_b if args.model_b else args.model)
    for p in (model_path_a, model_path_b):
        if not os.path.exists(p):
            sys.exit(f'Model not found: {p}')

    print(f'Model A:  {model_path_a}  (pos={pos_boost_a} bfs={bfs_boost_a} sims={sims_a})')
    print(f'Model B:  {model_path_b}  (pos={pos_boost_b} bfs={bfs_boost_b} sims={sims_b})')
    print(f'Threshold: Model B score >= {args.threshold} → Model B is competitive')
    print()

    # Serialize weights for both models
    buf_a = io.BytesIO()
    torch.save(load_model(model_path_a).state_dict(), buf_a)
    sd_bytes_a = buf_a.getvalue()

    buf_b = io.BytesIO()
    torch.save(load_model(model_path_b).state_dict(), buf_b)
    sd_bytes_b = buf_b.getvalue()

    # Pre-generate n_pairs opening sequences; each played twice (A first, B first)
    n_pairs = args.games // 2
    worker_args = []
    for pair_idx in range(n_pairs):
        state = State()
        opening_actions = []
        for _ in range(EN_FORCED_OPENING):
            if state.is_done():
                break
            action = int(np.random.choice(state.legal_actions()))
            opening_actions.append(action)
            state = state.next(action)
        for side, game_idx in enumerate([pair_idx * 2, pair_idx * 2 + 1]):
            worker_args.append((
                sd_bytes_a, sd_bytes_b, game_idx,
                pos_boost_a, bfs_boost_a,
                pos_boost_b, bfs_boost_b,
                sims_a, sims_b,
                EN_TEMPERATURE, EN_TEMP_CUTOFF,
                opening_actions,
            ))

    pool = mp.Pool()
    completed = wins_a = draws = wins_b = 0
    total_plies = 0

    for point_a, plies, all_actions, a_is_first in pool.imap_unordered(_worker, worker_args):
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
              f'Model A wins: {wins_a}  Draws: {draws}  Model B wins: {wins_b}  '
              f'| Model B score: {score_b:.3f}', end='')

    pool.close()
    pool.join()
    print()

    score_a = (wins_a + 0.5 * draws) / args.games
    score_b = (wins_b + 0.5 * draws) / args.games
    avg_plies = total_plies / args.games

    print('\n── Results ──────────────────────────────────────────────')
    print(f'  Model A ({os.path.basename(model_path_a)} sims={sims_a})  score: {score_a:.4f}  ({wins_a}W / {draws}D / {wins_b}L)')
    print(f'  Model B ({os.path.basename(model_path_b)} sims={sims_b})  score: {score_b:.4f}  ({wins_b}W / {draws}D / {wins_a}L)')
    print(f'  Avg game length:  {avg_plies:.1f} plies')
    print()
    if score_b >= args.threshold:
        print(f'✓ Model B score {score_b:.3f} >= {args.threshold}  →  Model B is competitive.')
    else:
        print(f'✗ Model B score {score_b:.3f} < {args.threshold}  →  Model A wins convincingly.')


if __name__ == '__main__':
    main()
