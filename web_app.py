# ====================
# AlphaQuoridor Web Dashboard
# ====================
# Single Flask server exposing:
#   /              — Training dashboard
#   /replay        — Eval game replay
#   /play          — Human vs AI
#   /inspect       — Position inspector
#   /logs          — Live log viewer
# API endpoints under /api/...

import os
import csv
import glob
import json
import time
from copy import deepcopy

import numpy as np
from flask import (
    Flask, render_template, jsonify, request, Response, session
)

# ── project imports ───────────────────────────────────────────────────────────
from config import LOGS_DIR, MODEL_DIR, USE_BFS_CHANNELS
from game import State, _get_blocked_edges, _bfs_goal_distances
from dual_network import load_model, DN_INPUT_SHAPE
from pv_mcts import pv_mcts_scores, predict, pv_mcts_full
import config as _cfg

# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'alphaquoridor-dashboard-2026'

# ── Model (loaded once at startup) ───────────────────────────────────────────
_model = None


def get_model():
    global _model
    if _model is None:
        best = os.path.join(MODEL_DIR, 'best.pt')
        latest = os.path.join(MODEL_DIR, 'latest.pt')
        path = best if os.path.exists(best) else (latest if os.path.exists(latest) else None)
        if path:
            print(f'[web_app] loading model from {path}')
            _model = load_model(path)
    return _model


# ── State helpers ─────────────────────────────────────────────────────────────

def _serialize_state(state: State) -> dict:
    return {
        'player': list(state.player),
        'enemy': list(state.enemy),
        'walls': list(state.walls),
        'depth': state.depth,
    }


def _deserialize_state(d: dict) -> State:
    return State(
        player=list(d['player']),
        enemy=list(d['enemy']),
        walls=list(d['walls']),
        depth=d['depth'],
    )


def _state_to_display(state: State) -> dict:
    """Convert internal state to an absolute (P1-perspective) display dict.

    Board convention:
      - Row 0 is top  (Player-2 starts here in absolute frame)
      - Row 6 is bottom (Player-1 starts here)
      - Absolute position = row * N + col, 0-indexed from top-left.

    Internal state stores positions in the *current-player's* frame where
    their own start is always row N-1 (bottom).  We unpack that below.
    """
    N = state.N
    is_fp = state.is_first_player()  # depth % 2 == 0 → P1's turn

    if is_fp:
        p1_pos        = state.player[0]           # P1 stored direct
        p2_pos        = N * N - 1 - state.enemy[0]  # P2 stored flipped
        p1_walls_left = state.player[1]
        p2_walls_left = state.enemy[1]
        walls_abs     = list(state.walls)         # already in P1 frame
    else:
        p1_pos        = state.enemy[0]            # P1 stored in enemy slot (direct)
        p2_pos        = N * N - 1 - state.player[0]  # P2 stored flipped in player slot
        p1_walls_left = state.enemy[1]
        p2_walls_left = state.player[1]
        walls_abs     = list(reversed(state.walls))  # undo rotation

    # Determine winner
    winner = None
    if state.is_done():
        if state.is_lose():
            winner = 'p2' if is_fp else 'p1'
        else:
            winner = 'draw'

    return {
        'N': N,
        'p1_pos': p1_pos,
        'p1_walls_left': p1_walls_left,
        'p2_pos': p2_pos,
        'p2_walls_left': p2_walls_left,
        'walls': walls_abs,       # 36-int list, P1-absolute frame
        'depth': state.depth,
        'is_p1_turn': is_fp,
        'legal': state.legal_actions(),
        'done': state.is_done(),
        'winner': winner,
    }


def _mcts_move(model, state, rollouts: int) -> int:
    """Run MCTS with the specified rollout budget and return the greedy action."""
    orig = _cfg.PV_EVALUATE_COUNT
    _cfg.PV_EVALUATE_COUNT = rollouts
    try:
        scores = pv_mcts_scores(model, deepcopy(state), temperature=0)
    finally:
        _cfg.PV_EVALUATE_COUNT = orig
    return int(state.legal_actions()[np.argmax(scores)])


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/replay')
def replay():
    return render_template('replay.html')


@app.route('/play')
def play_page():
    return render_template('play.html')


@app.route('/inspect')
def inspect_page():
    return render_template('inspect.html')


@app.route('/logs')
def logs_page():
    return render_template('logs.html')


# ── API: stats ────────────────────────────────────────────────────────────────

@app.route('/api/stats')
def api_stats():
    stats_path = os.path.join(LOGS_DIR, 'stats.csv')
    if not os.path.exists(stats_path):
        return jsonify([])
    rows = []
    with open(stats_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cleaned = {}
            for k, v in row.items():
                try:
                    cleaned[k] = float(v) if v != '' else None
                except ValueError:
                    cleaned[k] = v
            rows.append(cleaned)
    return jsonify(rows)


# ── API: eval games ───────────────────────────────────────────────────────────

@app.route('/api/eval_games')
def api_eval_games():
    games_dir = os.path.join(LOGS_DIR, 'eval_games')
    if not os.path.exists(games_dir):
        return jsonify([])
    files = sorted(glob.glob(os.path.join(games_dir, 'cycle_*.json')))
    result = []
    for fp in files:
        name = os.path.basename(fp)
        try:
            cycle = int(name.replace('cycle_', '').replace('.json', ''))
        except ValueError:
            continue
        result.append({'cycle': cycle, 'filename': name})
    return jsonify(result)


@app.route('/api/eval_games/<int:cycle>')
def api_eval_game(cycle: int):
    games_dir = os.path.join(LOGS_DIR, 'eval_games')
    path = os.path.join(games_dir, f'cycle_{cycle:04d}.json')
    if not os.path.exists(path):
        return jsonify({'error': 'not found'}), 404
    with open(path, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))


# ── API: human play ───────────────────────────────────────────────────────────

@app.route('/api/play/new', methods=['POST'])
def api_play_new():
    data = request.get_json() or {}
    rollouts = int(data.get('rollouts', 400))
    human_first = bool(data.get('human_first', True))
    vs_human = bool(data.get('vs_human', False))

    state = State()
    session['game'] = _serialize_state(state)
    session['rollouts'] = rollouts
    session['human_first'] = human_first
    session['vs_human'] = vs_human
    session['game_history'] = []

    display = _state_to_display(state)

    # If AI goes first (human is P2), play the AI move immediately
    if not human_first:
        model = get_model()
        if model:
            ai_action = _mcts_move(model, state, rollouts)
            state = state.next(ai_action)
            session['game'] = _serialize_state(state)
            display = _state_to_display(state)
            display['ai_action'] = ai_action
        else:
            display['ai_action'] = None
    else:
        display['ai_action'] = None

    display['can_undo'] = False
    return jsonify(display)


@app.route('/api/play/move', methods=['POST'])
def api_play_move():
    data = request.get_json() or {}
    action = int(data.get('action', -1))

    game_data = session.get('game')
    if game_data is None:
        return jsonify({'error': 'no active game'}), 400

    # Push current state onto undo history BEFORE applying the move
    history = session.get('game_history', [])
    history.append(game_data)
    session['game_history'] = history

    state = _deserialize_state(game_data)
    legal = state.legal_actions()

    if action not in legal:
        history.pop()
        session['game_history'] = history
        return jsonify({'error': f'illegal action {action}'}), 400

    state = state.next(action)
    display = _state_to_display(state)
    display['ai_action'] = None

    if state.is_done() or session.get('vs_human', False):
        session['game'] = _serialize_state(state)
        display['can_undo'] = len(history) > 0
        return jsonify(display)

    # AI responds
    model = get_model()
    rollouts = session.get('rollouts', 400)
    if model:
        ai_action = _mcts_move(model, state, rollouts)
        state = state.next(ai_action)
        display = _state_to_display(state)
        display['ai_action'] = ai_action
    else:
        display['ai_action'] = None

    session['game'] = _serialize_state(state)
    display['can_undo'] = len(history) > 0
    return jsonify(display)


@app.route('/api/play/undo', methods=['POST'])
def api_play_undo():
    history = session.get('game_history', [])
    if not history:
        return jsonify({'error': 'nothing to undo'}), 400
    prev_state_data = history.pop()
    session['game_history'] = history
    session['game'] = prev_state_data
    display = _state_to_display(_deserialize_state(prev_state_data))
    display['can_undo'] = len(history) > 0
    display['ai_action'] = None
    return jsonify(display)


# ── API: position inspector ───────────────────────────────────────────────────

@app.route('/api/inspect', methods=['POST'])
def api_inspect():
    data = request.get_json() or {}
    player = data.get('player', [45, 5])
    enemy  = data.get('enemy',  [45, 5])
    walls  = data.get('walls',  [0] * 36)
    depth  = int(data.get('depth', 0))
    mcts_rollouts = int(data.get('mcts_rollouts', 0))

    state = State(
        player=list(player),
        enemy=list(enemy),
        walls=list(walls),
        depth=depth,
    )
    N = state.N
    n_channels = DN_INPUT_SHAPE[2]

    # Input channels — shape (C, H, W)
    pieces_raw = state.pieces_array()
    channels_np = np.array(pieces_raw, dtype=np.float32).reshape(n_channels, N, N)
    channels = [channels_np[c].tolist() for c in range(n_channels)]

    # Per-cell BFS distances (always computed, even if not in input channels)
    walls_t   = tuple(state.walls)
    walls_rot = tuple(reversed(walls_t))

    h, v       = _get_blocked_edges(N, walls_t)
    p_bfs_flat = _bfs_goal_distances(N, h, v, 0)          # player goal = row 0

    h_r, v_r         = _get_blocked_edges(N, walls_rot)
    e_bfs_flat_own   = _bfs_goal_distances(N, h_r, v_r, 0)  # enemy goal = row 0 in their frame
    e_bfs_flat_abs   = list(reversed(e_bfs_flat_own))        # flip to absolute coords

    def to_grid(flat):
        return [flat[r * N: (r + 1) * N] for r in range(N)]

    # Legal actions
    legal = state.legal_actions()

    # Model forward pass
    model = get_model()
    policies = None
    value    = None
    if model is not None:
        pol_arr, value = predict(model, state)
        policies = {int(a): float(p) for a, p in zip(legal, pol_arr)}

    # Optional MCTS forward pass
    mcts_policies = None
    mcts_value    = None
    if mcts_rollouts > 0 and model is not None:
        visit_probs, root_q = pv_mcts_full(model, deepcopy(state), mcts_rollouts)
        mcts_policies = {int(a): float(p) for a, p in zip(legal, visit_probs)}
        mcts_value    = root_q

    return jsonify({
        'channels':      channels,
        'p_bfs':         to_grid(p_bfs_flat),
        'e_bfs':         to_grid(e_bfs_flat_abs),
        'legal_actions': legal,
        'policies':      policies,
        'value':         value,
        'mcts_policies': mcts_policies,
        'mcts_value':    mcts_value,
        'N':             N,
        'n_channels':    n_channels,
        'use_bfs':       USE_BFS_CHANNELS,
    })


# ── API: live log SSE ─────────────────────────────────────────────────────────

@app.route('/api/logs/stream')
def api_logs_stream():
    def generate():
        log_files = sorted(glob.glob(os.path.join(LOGS_DIR, '*.log')))
        if not log_files:
            yield 'data: {"text": "[no log files found]"}\n\n'
            return

        log_path = log_files[-1]

        # Send last 500 lines as a backfill
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        for line in lines[-500:]:
            payload = json.dumps({'text': line.rstrip(), 'backfill': True})
            yield f'data: {payload}\n\n'

        # Tail the file indefinitely
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    payload = json.dumps({'text': line.rstrip(), 'backfill': False})
                    yield f'data: {payload}\n\n'
                else:
                    time.sleep(0.4)
                    yield ': heartbeat\n\n'

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Eagerly warm up the model so the first /play request isn't slow
    print('[web_app] warming up model...')
    get_model()
    app.run(debug=False, port=5000, threaded=True)
