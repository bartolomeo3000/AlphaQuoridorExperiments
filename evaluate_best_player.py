# ====================
# Evaluation of Best Player
# ====================

# Import packages
from game import State, random_action
from pv_mcts import pv_mcts_action
import os
from dual_network import load_model
from config import EP_RANDOM_GAMES, EP_GREEDY_GAMES, EP_BFS_GAMES, MODEL_DIR
from collections import deque
import numpy as np
import random

from config import EP_RANDOM_GAMES, EP_GREEDY_GAMES, EP_BFS_GAMES

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

# Evaluation of any algorithm
def evaluate_algorithm_of(label, next_actions, game_count):
    if game_count == 0:
        return None
    total_point = 0
    wins = draws = losses = 0
    for i in range(game_count):
        if i % 2 == 0:
            p = play(next_actions)
        else:
            p = 1 - play(list(reversed(next_actions)))
        total_point += p
        if p == 1.0:   wins   += 1
        elif p == 0.0: losses += 1
        else:          draws  += 1
        print(f'\r{label} {i+1}/{game_count}  W:{wins} D:{draws} L:{losses}', end='')
    print('')
    average_point = total_point / game_count
    print(f'{label} — W:{wins}  D:{draws}  L:{losses}  Score:{average_point:.2f}')
    return round(average_point, 2)

# Evaluation of the best player
def evaluate_best_player():
    # Load the model of the best player
    model = load_model(os.path.join(MODEL_DIR, 'best.pt'))

    # Generate a function to select actions using PV MCTS
    next_pv_mcts_action = pv_mcts_action(model, 0.0)

    # VS Random (10 games)
    next_actions = (next_pv_mcts_action, random_action)
    vs_random = evaluate_algorithm_of('VS_Random', next_actions, EP_RANDOM_GAMES)

    # VS Greedy Forward — naive: always steps to lowest reachable row (20 games)
    next_actions = (next_pv_mcts_action, greedy_forward_action)
    vs_greedy = evaluate_algorithm_of('VS_GreedyForward', next_actions, EP_GREEDY_GAMES)

    # VS BFS Forward — optimal pawn runner: follows true shortest path through walls (20 games)
    next_actions = (next_pv_mcts_action, bfs_forward_action)
    vs_bfs = evaluate_algorithm_of('VS_BFS', next_actions, EP_BFS_GAMES)

    # Clear model
    del model

    return {'vs_random': vs_random, 'vs_greedy': vs_greedy, 'vs_bfs': vs_bfs}

# Operation check
if __name__ == '__main__':
    evaluate_best_player()
