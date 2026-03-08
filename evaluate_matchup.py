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

from game import State
from dual_network import DualNetwork, load_model
from config import (
    MODEL_DIR, EN_TEMPERATURE, EN_TEMP_CUTOFF, EN_FORCED_OPENING,
    POSITION_PRIOR_BOOST, BFS_MOVE_BOOST, BFS_MOVE_PENALTY, BFS_ADVANCE_FLOOR,
    BFS_PUCT_RETREAT_PENALTY, BFS_PUCT_ADVANCE_BONUS, BFS_WALL_PUCT_SCALE,
    PV_EVALUATE_COUNT,
)
from pv_mcts import pv_mcts_scores


# ── Worker (must be top-level for multiprocessing spawn on Windows) ───────────

def _worker(args):
    sd_bytes_a, sd_bytes_b, game_idx, \
        pos_boost_a, bfs_boost_a, bfs_penalty_a, bfs_floor_a, bfs_retreat_a, bfs_wall_a, bfs_advance_a, \
        pos_boost_b, bfs_boost_b, bfs_penalty_b, bfs_floor_b, bfs_retreat_b, bfs_wall_b, bfs_advance_b, \
        sims_a, sims_b, temperature, temp_cutoff, opening_actions = args

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
            scores = pv_mcts_scores(model_a, deepcopy(state), t,
                sims=sims_a, pos_boost=pos_boost_a, bfs_boost=bfs_boost_a,
                bfs_penalty=bfs_penalty_a, bfs_floor=bfs_floor_a,
                bfs_retreat_penalty=bfs_retreat_a, bfs_wall_scale=bfs_wall_a, bfs_advance_bonus=bfs_advance_a)
        else:
            scores = pv_mcts_scores(model_b, deepcopy(state), t,
                sims=sims_b, pos_boost=pos_boost_b, bfs_boost=bfs_boost_b,
                bfs_penalty=bfs_penalty_b, bfs_floor=bfs_floor_b,
                bfs_retreat_penalty=bfs_retreat_b, bfs_wall_scale=bfs_wall_b, bfs_advance_bonus=bfs_advance_b)
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

    pos_a      = cfg.get('pos_a',      POSITION_PRIOR_BOOST)
    bfs_a      = cfg.get('bfs_a',      BFS_MOVE_BOOST)
    penalty_a  = cfg.get('penalty_a',  BFS_MOVE_PENALTY)
    floor_a    = cfg.get('floor_a',    BFS_ADVANCE_FLOOR)
    retreat_a  = cfg.get('retreat_a',  BFS_PUCT_RETREAT_PENALTY)
    wall_a     = cfg.get('wall_a',     BFS_WALL_PUCT_SCALE)
    advance_a  = cfg.get('advance_a',  BFS_PUCT_ADVANCE_BONUS)
    sims_a     = cfg.get('sims_a',     PV_EVALUATE_COUNT)
    pos_b      = cfg.get('pos_b',      POSITION_PRIOR_BOOST)
    bfs_b      = cfg.get('bfs_b',      BFS_MOVE_BOOST)
    penalty_b  = cfg.get('penalty_b',  BFS_MOVE_PENALTY)
    floor_b    = cfg.get('floor_b',    BFS_ADVANCE_FLOOR)
    retreat_b  = cfg.get('retreat_b',  BFS_PUCT_RETREAT_PENALTY)
    wall_b     = cfg.get('wall_b',     BFS_WALL_PUCT_SCALE)
    advance_b  = cfg.get('advance_b',  BFS_PUCT_ADVANCE_BONUS)
    sims_b     = cfg.get('sims_b',     PV_EVALUATE_COUNT)

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
                pos_a, bfs_a, penalty_a, floor_a, retreat_a, wall_a, advance_a,
                pos_b, bfs_b, penalty_b, floor_b, retreat_b, wall_b, advance_b,
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
                pos_boost_a, bfs_boost_a, BFS_MOVE_PENALTY, BFS_ADVANCE_FLOOR, BFS_PUCT_RETREAT_PENALTY, BFS_WALL_PUCT_SCALE, BFS_PUCT_ADVANCE_BONUS,
                pos_boost_b, bfs_boost_b, BFS_MOVE_PENALTY, BFS_ADVANCE_FLOOR, BFS_PUCT_RETREAT_PENALTY, BFS_WALL_PUCT_SCALE, BFS_PUCT_ADVANCE_BONUS,
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
