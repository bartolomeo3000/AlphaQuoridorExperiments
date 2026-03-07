# ====================
# New Parameter Evaluation Section
# ====================

# Import packages
import os
from game import State
from pv_mcts import pv_mcts_action, pv_mcts_scores
from dual_network import load_model, DualNetwork, DN_OUTPUT_SIZE
from config import (
    EN_GAME_COUNT, EN_TEMPERATURE, EN_TEMP_CUTOFF, EN_FORCED_OPENING,
    EN_PROMOTE_THRESHOLD, EN_DRAW_DISTANCE_SCORING, DRAW_SHAPE_SCALE,
    MODEL_DIR, USE_BFS_CHANNELS, OPENING_DEPTH, LOGS_DIR,
)
from shutil import copy
from copy import deepcopy
from self_play import write_data
import numpy as np
import io
import json
import os
import torch
import multiprocessing as mp
from collections import Counter

# Persistent pool — created once, reused across all evaluation calls
_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        _pool = mp.Pool(processes=max(1, mp.cpu_count() - 1))
    return _pool

# Points for the first player

def first_player_point(ended_state):
    # 1: first player wins, 0: first player loses, 0.5: draw
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    if EN_DRAW_DISTANCE_SCORING:
        N = ended_state.N
        if USE_BFS_CHANNELS:
            p_dist, e_dist = ended_state.bfs_distances()
        else:
            p_dist = ended_state.player[0] // N
            e_dist = ended_state.enemy[0] // N
        draw_val = DRAW_SHAPE_SCALE * (e_dist - p_dist) / (N - 1)
        if not ended_state.is_first_player():
            draw_val = -draw_val
        return 0.5 + draw_val
    return 0.5

# Execute one game
def play(next_actions):
    # Generate state
    state = State()

    # Loop until the game ends
    while True:
        # When the game ends
        if state.is_done():
            break

        # Get action
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # Get the next state
        state = state.next(action)

    # Return points for the first player
    return first_player_point(state)

# Replace the best player
def update_best_player():
    copy(os.path.join(MODEL_DIR, 'latest.pt'), os.path.join(MODEL_DIR, 'best.pt'))
    print('Change BestPlayer')


# Worker: loads both models on CPU and plays one game.
# Returns (point_for_model0, game_history) where game_history is
# a list of (pieces_array, policy_vector, value) tuples suitable for training.
def _eval_worker(args):
    sd0_bytes, sd1_bytes, game_idx, opening_actions = args

    model0 = DualNetwork()
    model0.load_state_dict(torch.load(io.BytesIO(sd0_bytes), map_location='cpu'))
    model0.eval()

    model1 = DualNetwork()
    model1.load_state_dict(torch.load(io.BytesIO(sd1_bytes), map_location='cpu'))
    model1.eval()

    # model0 plays first in even games, second in odd games
    if game_idx % 2 == 0:
        models = [model0, model1]
    else:
        models = [model1, model0]

    state = State()
    N = state.N
    history = []
    draw_values = []
    move_count = 0
    opening = []
    actions = []

    # Replay the pre-generated opening (same sequence for both games in a pair;
    # game_idx even = model0 first, odd = model1 first — bias cancels within pair).
    for action in opening_actions:
        if state.is_done():
            break
        state = state.next(action)
        actions.append(action)
        move_count += 1
        if move_count <= OPENING_DEPTH:
            opening.append(action)

    while True:
        if state.is_done():
            break

        model = models[0] if state.is_first_player() else models[1]
        t = EN_TEMPERATURE if move_count < EN_TEMP_CUTOFF else 0.0

        scores = pv_mcts_scores(model, deepcopy(state), t, add_noise=False)

        # Record state and policy distribution for training
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([state.pieces_array(), policies, None])

        # Draw shaping value (zero-sum, same formula as self-play)
        if USE_BFS_CHANNELS:
            p_dist, e_dist = state.bfs_distances()
        else:
            p_dist = state.player[0] // N
            e_dist = state.enemy[0] // N
        draw_values.append(DRAW_SHAPE_SCALE * (e_dist - p_dist) / (N - 1))

        action = np.random.choice(state.legal_actions(), p=scores)
        actions.append(int(action))
        if move_count < OPENING_DEPTH:
            opening.append(int(action))
        state = state.next(action)
        move_count += 1

    # Assign training values
    if state.is_draw():
        for i in range(len(history)):
            history[i][2] = draw_values[i]
    else:
        value = -1 if state.is_first_player() else 1
        for i in range(len(history)):
            history[i][2] = value
            value = -value

    # Compute point for model0
    point = first_player_point(state)
    if game_idx % 2 != 0:
        point = 1 - point

    return point, history, tuple(opening), actions, game_idx


# Network evaluation
def evaluate_network(cycle_num=None):
    # Serialize both models to bytes so workers can load them on CPU
    def to_bytes(path):
        model = load_model(path)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        del model
        return buf.getvalue()

    sd0_bytes = to_bytes(os.path.join(MODEL_DIR, 'latest.pt'))
    sd1_bytes = to_bytes(os.path.join(MODEL_DIR, 'best.pt'))

    # Pre-generate EN_GAME_COUNT/2 opening sequences; each is played twice
    # (model0 first, then model1 first) so any positional advantage cancels.
    n_pairs = EN_GAME_COUNT // 2
    args = []
    for pair_idx in range(n_pairs):
        state = State()
        opening_actions = []
        for _ in range(EN_FORCED_OPENING):
            if state.is_done():
                break
            action = int(np.random.choice(state.legal_actions()))
            opening_actions.append(action)
            state = state.next(action)
        args.append((sd0_bytes, sd1_bytes, pair_idx * 2,     opening_actions))  # model0 first
        args.append((sd0_bytes, sd1_bytes, pair_idx * 2 + 1, opening_actions))  # model1 first

    print(f'Evaluate: {EN_GAME_COUNT} games ({n_pairs} paired openings) across workers...')
    pool = _get_pool()
    results = []
    points = []
    eval_history = []
    opening_seqs = []
    game_records = []
    completed = 0
    wins_live = draws_live = losses_live = 0
    for point, game_history, opening_seq, actions, gidx in pool.imap_unordered(_eval_worker, args):
        results.append((point, game_history))
        points.append(point)
        eval_history.extend(game_history)
        opening_seqs.append(opening_seq)
        game_records.append({
            'actions':      actions,
            'latest_first': (gidx % 2 == 0),
            'result':       'latest_win' if point > 0.6 else ('best_win' if point < 0.4 else 'draw'),
        })
        completed += 1
        if point > 0.6:   wins_live   += 1
        elif point < 0.4: losses_live += 1
        else:             draws_live  += 1
        print(f'\rEvaluate {completed}/{EN_GAME_COUNT}  '
              f'W:{wins_live}  D:{draws_live}  L:{losses_live}', end='')
    print('')

    # Save eval games to replay buffer — same format as self-play data
    write_data(eval_history)
    print(f'Saved {len(eval_history)} eval positions to replay buffer')

    # Save eval game records (action sequences) for later replay
    games_dir = os.path.join(LOGS_DIR, 'eval_games')
    os.makedirs(games_dir, exist_ok=True)
    if cycle_num is not None:
        games_path = os.path.join(games_dir, f'cycle_{cycle_num:04d}.json')
    else:
        from datetime import datetime
        games_path = os.path.join(games_dir, datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    with open(games_path, 'w', encoding='utf-8') as f:
        json.dump({'cycle': cycle_num, 'games': game_records}, f)
    print(f'Saved {len(game_records)} eval game records to {games_path}')

    wins   = sum(1 for p in points if p > 0.6)
    losses = sum(1 for p in points if p < 0.4)
    draws  = EN_GAME_COUNT - wins - losses
    average_point = sum(points) / EN_GAME_COUNT
    print(f'Evaluation result — '
          f'New wins: {wins}  Draws: {draws}  New losses: {losses}  '
          f'({EN_GAME_COUNT} games)')
    print(f'New-model score vs best: {average_point:.4f}  '
          f'({"PROMOTED" if average_point >= EN_PROMOTE_THRESHOLD else "rejected"})')

    # Opening diversity stats
    counts = Counter(opening_seqs)
    n_unique = len(counts)
    freqs = np.array(list(counts.values()), dtype=np.float32) / EN_GAME_COUNT
    en_entropy = float(-np.sum(freqs * np.log2(freqs + 1e-12)))
    top1_count = counts.most_common(1)[0][1] if counts else 0
    top3 = counts.most_common(3)
    top3_str = '  '.join(f'{cnt}x{list(seq)}' for seq, cnt in top3)
    print(f'Eval opening diversity (depth={OPENING_DEPTH}) — '
          f'unique:{n_unique}/{EN_GAME_COUNT}  entropy:{en_entropy:.2f} bits  '
          f'top3: {top3_str}')

    stats = {
        'score':    round(average_point, 4),
        'wins':     wins, 'draws': draws, 'losses': losses,
        'entropy':  round(en_entropy, 2),
        'unique':   n_unique,
        'top1_count': top1_count,
    }

    # Replace the best player
    if average_point >= EN_PROMOTE_THRESHOLD:
        update_best_player()
        return True, stats
    else:
        return False, stats

# Operation check
if __name__ == '__main__':
    evaluate_network()
