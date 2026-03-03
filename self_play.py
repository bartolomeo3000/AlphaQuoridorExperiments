# ====================
# Self-Play Part
# ====================

# Importing packages
from game import State
from pv_mcts import pv_mcts_scores
from dual_network import DN_OUTPUT_SIZE, load_model, DualNetwork
from datetime import datetime
from pathlib import Path
from collections import Counter
import numpy as np
import pickle
import os
import io
import time
import torch
import multiprocessing as mp
from copy import deepcopy

from config import (
    SP_GAME_COUNT, SP_TEMPERATURE, TEMP_CUTOFF, SP_NUM_WORKERS, OPENING_DEPTH,
    DRAW_SHAPE_SCALE, MODEL_DIR, DATA_DIR, SP_CKPT_EVERY,
)

# Persistent pool — created once, reused across all training cycles so workers
# only pay the Python/PyTorch import cost once per run (not once per cycle).
_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        _pool = mp.Pool(processes=SP_NUM_WORKERS)
    return _pool

# Value of the first player — only called on win/loss (draw is handled separately)
def first_player_value(ended_state):
    # 1: First player wins, -1: First player loses
    return -1 if ended_state.is_first_player() else 1

# Saving training data
def write_data(history):
    now = datetime.now()
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, '{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second))
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# Atomically save self-play checkpoint so a mid-game kill loses at most
# SP_CKPT_EVERY games rather than the entire cycle's worth.
def _save_ckpt(path, history, outcomes, opening_seqs, completed):
    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, 'wb') as f:
        pickle.dump({
            'history':      history,
            'outcomes':     outcomes,
            'opening_seqs': opening_seqs,
            'completed':    completed,
        }, f)
    # On Windows, Defender/Search Indexer can briefly lock the destination file
    # causing os.replace to raise PermissionError.  Retry a few times.
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt < 4:
                time.sleep(0.3)
            else:
                # Give up on rename; remove the orphaned .tmp so it doesn't linger
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise

# Executing one game
def play(model):
    # Training data
    history = []
    draw_values = []  # per-step draw value (for draw shaping)
    opening = []      # first OPENING_DEPTH actions for diversity tracking

    # Generating the state
    state = State()
    N = state.N
    move_num = 0

    while True:
        # When the game ends
        if state.is_done():
            break

        # Temperature schedule: explore early, play best move later
        temperature = SP_TEMPERATURE if move_num < TEMP_CUTOFF else 0.0

        # Getting the probability distribution of legal moves (with Dirichlet noise)
        scores = pv_mcts_scores(model, deepcopy(state), temperature, add_noise=True)

        # Adding the state and policy to the training data
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([state.pieces_array(), policies, None])

        # Draw shaping: value = DRAW_SHAPE_SCALE * (enemy_dist - player_dist) / (N-1)
        # Range [-DRAW_SHAPE_SCALE, +DRAW_SHAPE_SCALE]. Zero-sum: after board rotation enemy_dist and player_dist swap.
        p_row = state.player[0] // N  # 0 = at goal, N-1 = at start
        e_row = state.enemy[0] // N   # 0 = enemy at goal (= player loses), N-1 = enemy at start
        draw_values.append(DRAW_SHAPE_SCALE * (e_row - p_row) / (N - 1))

        # Getting the action
        action = np.random.choice(state.legal_actions(), p=scores)

        # Track opening moves for diversity stats
        if move_num < OPENING_DEPTH:
            opening.append(int(action))

        # Getting the next state
        state = state.next(action)
        move_num += 1

    # Adding the value to the training data
    if state.is_draw():
        outcome = 'D'
        for i in range(len(history)):
            history[i][2] = draw_values[i]
    else:
        value = first_player_value(state)
        outcome = 'L' if value == -1 else 'W'  # from first player's perspective
        for i in range(len(history)):
            history[i][2] = value
            value = -value
    return history, outcome, tuple(opening)

# Worker function: plays exactly one game. Defined at module level so it can
# be pickled on Windows (spawn). Model weights are passed as bytes each call
# so the pool can be reused across cycles with an updated model.
def _worker(args):
    state_dict_bytes, = args

    # Load model onto CPU inside the worker
    model = DualNetwork()
    model.load_state_dict(torch.load(io.BytesIO(state_dict_bytes), map_location='cpu'))
    model.eval()

    history, outcome, opening_seq = play(model)
    return history, outcome, opening_seq

# Self-Play
def self_play(cycle_num=None):
    """Run self-play for one cycle.

    Args:
        cycle_num: When given, a checkpoint is written every SP_CKPT_EVERY games
                   so the cycle can be resumed after a kill without losing much
                   progress.  Pass None (default) to skip checkpointing (e.g.
                   when calling self_play() directly from __main__).
    """
    # Checkpoint path for this cycle (None → no checkpointing)
    ckpt_path = None
    if cycle_num is not None:
        ckpt_path = os.path.join(DATA_DIR, f'.sp_ckpt_c{cycle_num:04d}.pkl')

    # ── Load partial checkpoint if one exists ────────────────────────────────
    history      = []
    outcomes     = {'W': 0, 'D': 0, 'L': 0}
    opening_seqs = []
    completed    = 0

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, 'rb') as f:
                ckpt = pickle.load(f)
            history      = ckpt['history']
            outcomes     = ckpt['outcomes']
            opening_seqs = ckpt['opening_seqs']
            completed    = ckpt['completed']
            print(f'[resume] self-play checkpoint found: '
                  f'{completed}/{SP_GAME_COUNT} games already done')
        except Exception as exc:
            print(f'[resume] checkpoint unreadable ({exc}), starting fresh')
            history, outcomes, opening_seqs, completed = [], {'W':0,'D':0,'L':0}, [], 0

    # ── Serialize model weights once; send to all workers via IPC ─────────────
    gpu_model = load_model(os.path.join(MODEL_DIR, 'best.pt'))
    buf = io.BytesIO()
    torch.save(gpu_model.state_dict(), buf)
    state_dict_bytes = buf.getvalue()
    del gpu_model

    # ── Play the remaining games ──────────────────────────────────────────────
    remaining = SP_GAME_COUNT - completed
    if remaining > 0:
        args = [(state_dict_bytes,)] * remaining
        pool = _get_pool()
        since_ckpt = 0
        for game_history, outcome, opening_seq in pool.imap_unordered(_worker, args):
            history.extend(game_history)
            outcomes[outcome] += 1
            opening_seqs.append(opening_seq)
            completed += 1
            since_ckpt += 1
            print(f'\rSelf-play {completed}/{SP_GAME_COUNT}  '
                  f'W:{outcomes["W"]}  D:{outcomes["D"]}  L:{outcomes["L"]}', end='')
            if ckpt_path and since_ckpt >= SP_CKPT_EVERY:
                try:
                    _save_ckpt(ckpt_path, history, outcomes, opening_seqs, completed)
                except Exception as exc:
                    print(f'\n[ckpt] WARNING: save failed at game {completed} ({exc}) — continuing without checkpoint')
                since_ckpt = 0
    else:
        print(f'\rSelf-play {completed}/{SP_GAME_COUNT}  '
              f'W:{outcomes["W"]}  D:{outcomes["D"]}  L:{outcomes["L"]}', end='')

    total = SP_GAME_COUNT
    print(f'\nSelf-play done — '
          f'W:{outcomes["W"]} ({100*outcomes["W"]/total:.0f}%)  '
          f'D:{outcomes["D"]} ({100*outcomes["D"]/total:.0f}%)  '
          f'L:{outcomes["L"]} ({100*outcomes["L"]/total:.0f}%)  '
          f'| {len(history)} positions')

    # Opening diversity stats
    counts = Counter(opening_seqs)
    n_unique = len(counts)
    freqs = np.array(list(counts.values()), dtype=np.float32) / total
    entropy = float(-np.sum(freqs * np.log2(freqs + 1e-12)))
    top3 = counts.most_common(3)
    top3_str = '  '.join(f'{cnt}x{list(seq)}' for seq, cnt in top3)
    print(f'Opening diversity (depth={OPENING_DEPTH}) — '
          f'unique:{n_unique}/{total}  entropy:{entropy:.2f} bits  '
          f'top3: {top3_str}')

    # Saving the training data — delete the checkpoint only AFTER a successful
    # write so a kill during pickle.dump doesn't lose all games.
    write_data(history)
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return {
        'W_pct':     100 * outcomes['W'] / total,
        'D_pct':     100 * outcomes['D'] / total,
        'L_pct':     100 * outcomes['L'] / total,
        'positions': len(history),
        'unique':    n_unique,
        'entropy':   round(entropy, 2),
        'top1_count': top3[0][1] if top3 else 0,
    }

# Running the function
if __name__ == '__main__':
    self_play()
