# ====================
# Evaluation of Best Player
# ====================

# Import packages
from game import State, random_action
from pv_mcts import pv_mcts_action
import os
import json
from dual_network import load_model
from config import EP_RANDOM_GAMES, EP_GREEDY_GAMES, EP_BFS_GAMES, MODEL_DIR, EN_FORCED_OPENING, LOGS_DIR
from collections import deque
import numpy as np
import random

# Points for the first player
def first_player_point(ended_state):
    # 1: first player wins, 0: first player loses, 0.5: draw
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# Greedy forward agent: never places walls, always moves toward goal (row 0).
# Prefers the legal move with the lowest resulting row; breaks ties randomly.
def greedy_forward_action(state):
    N = state.N
    # Only consider position moves (no walls)
    pos_moves = [a for a in state.legal_actions() if a < N * N]
    if not pos_moves:
        # Fallback: should never happen but take any legal action
        return random.choice(state.legal_actions())
    # Pick the move that lands on the lowest row (closest to goal = row 0)
    best_row = min(a // N for a in pos_moves)
    best_moves = [a for a in pos_moves if a // N == best_row]
    return random.choice(best_moves)


# BFS agent: never places walls, always takes the next step on the true shortest
# path to row 0 through the current wall+opponent graph.
# Uses state.legal_actions_pos() as the transition function so all Quoridor
# movement rules (jumping over opponent, diagonal when blocked) are respected.
# The opponent position is fixed during BFS (agent ignores opponent strategy).
def bfs_forward_action(state):
    N = state.N
    start = state.player[0]

    if start // N == 0:
        return random.choice(state.legal_actions())

    # BFS on player positions with enemy held fixed.
    # legal_actions_pos(pos) handles walls + all jump/diagonal rules.
    parent = {start: None}
    queue = deque([start])
    goal_pos = None

    while queue and goal_pos is None:
        pos = queue.popleft()
        for nb in state.legal_actions_pos(pos):
            if nb not in parent:
                parent[nb] = pos
                if nb // N == 0:
                    goal_pos = nb
                    break
                queue.append(nb)

    if goal_pos is None:
        # Completely walled off — fall back to greedy
        return greedy_forward_action(state)

    # Trace back to find the immediate next step from start
    step = goal_pos
    while parent[step] != start:
        step = parent[step]

    return step

# Execute one game — returns (first_player_point, list_of_all_actions)
def play(next_actions, opening_actions=()):
    # Generate state
    state = State()
    actions = list(opening_actions)

    # Replay forced opening before handing off to the agents
    for action in opening_actions:
        if state.is_done():
            break
        state = state.next(action)

    # Loop until the game ends
    while True:
        # When the game ends
        if state.is_done():
            break

        # Get action
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = int(next_action(state))
        actions.append(action)

        # Get the next state
        state = state.next(action)

    # Return points for the first player and the full action list
    return first_player_point(state), actions

# Evaluation of any algorithm — returns (average_point, game_records)
# nn_action_factory: callable() -> action_fn, called fresh for every game
def evaluate_algorithm_of(label, nn_action_factory, opponent_action, game_count):
    if game_count == 0:
        return None, []
    total_point = 0
    wins = draws = losses = 0
    game_records = []
    n_pairs = game_count // 2
    for pair_idx in range(n_pairs):
        # Pre-generate one opening for both games in this pair
        state = State()
        opening_actions = []
        for _ in range(EN_FORCED_OPENING):
            if state.is_done():
                break
            action = int(np.random.choice(state.legal_actions()))
            opening_actions.append(action)
            state = state.next(action)

        for side in range(2):
            i = pair_idx * 2 + side
            nn_action = nn_action_factory()  # fresh move_count per game
            if side == 0:
                p, actions = play((nn_action, opponent_action), opening_actions)
                nn_first = True
            else:
                p_raw, actions = play((opponent_action, nn_action), opening_actions)
                p = 1 - p_raw
                nn_first = False
            total_point += p
            if p == 1.0:   wins   += 1; result = 'nn_win'
            elif p == 0.0: losses += 1; result = 'opponent_win'
            else:          draws  += 1; result = 'draw'
            game_records.append({'actions': actions, 'nn_first': nn_first, 'result': result})
            print(f'\r{label} {i+1}/{game_count}  W:{wins} D:{draws} L:{losses}', end='')
    print('')
    average_point = total_point / game_count
    print(f'{label} — W:{wins}  D:{draws}  L:{losses}  Score:{average_point:.2f}')
    return round(average_point, 2), game_records

def _save_bench_games(cycle_num, opponent, score, records):
    if not records:
        return
    bench_dir = os.path.join(LOGS_DIR, 'bench_games')
    os.makedirs(bench_dir, exist_ok=True)
    if cycle_num is not None:
        fname = f'cycle_{cycle_num:04d}_vs_{opponent}.json'
    else:
        from datetime import datetime
        fname = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_vs_{opponent}.json'
    path = os.path.join(bench_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'cycle': cycle_num, 'opponent': opponent, 'score': score, 'games': records}, f)
    print(f'Saved {len(records)} bench game records (vs {opponent}) to {path}')


# Evaluation of the best player
def evaluate_best_player(cycle_num=None):
    # Load the model of the best player
    model = load_model(os.path.join(MODEL_DIR, 'best.pt'))

    # Generate a factory that produces a fresh pv_mcts_action per game
    # temperature=1 for first 8 model plies, greedy thereafter — adds variety across bench games
    def nn_factory():
        return pv_mcts_action(model, temperature=1.0, temp_cutoff=8)

    # VS Random
    vs_random, records_random = evaluate_algorithm_of('VS_Random', nn_factory, random_action, EP_RANDOM_GAMES)

    # VS Greedy Forward
    vs_greedy, records_greedy = evaluate_algorithm_of('VS_GreedyForward', nn_factory, greedy_forward_action, EP_GREEDY_GAMES)

    # VS BFS Forward
    vs_bfs, records_bfs = evaluate_algorithm_of('VS_BFS', nn_factory, bfs_forward_action, EP_BFS_GAMES)

    # Clear model
    del model

    # Save game records
    _save_bench_games(cycle_num, 'random', vs_random, records_random)
    _save_bench_games(cycle_num, 'greedy', vs_greedy, records_greedy)
    _save_bench_games(cycle_num, 'bfs',    vs_bfs,    records_bfs)

    return {'vs_random': vs_random, 'vs_greedy': vs_greedy, 'vs_bfs': vs_bfs}

# Operation check
if __name__ == '__main__':
    evaluate_best_player()
