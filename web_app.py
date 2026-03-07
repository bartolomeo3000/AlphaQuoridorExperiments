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
import threading
import subprocess
import sys
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

# ── Training subprocess ───────────────────────────────────────────────────────
_train_proc = None

def _stop_flag_path():
    return os.path.join(LOGS_DIR, '.stop_requested')

def _training_running():
    return _train_proc is not None and _train_proc.poll() is None

# ── Matchup state (runs in a background thread) ──────────────────────────
_matchup_lock   = threading.Lock()
_matchup_cancel = threading.Event()
_matchup_state  = {
    'status':    'idle',   # idle | running | done | cancelled
    'cfg':       None,
    'wins_a':    0,
    'draws':     0,
    'wins_b':    0,
    'completed': 0,
    'total':     0,
    'score_a':   0.0,
    'score_b':   0.0,
    'games':     [],       # list of {actions, a_first, result, plies}
}

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

    # BFS distances — always in absolute (P1/P2) terms
    p_dist, e_dist = state.bfs_distances()
    p1_bfs_dist = p_dist if is_fp else e_dist
    p2_bfs_dist = e_dist if is_fp else p_dist

    return {
        'N': N,
        'p1_pos': p1_pos,
        'p1_walls_left': p1_walls_left,
        'p1_bfs_dist': p1_bfs_dist,
        'p2_pos': p2_pos,
        'p2_walls_left': p2_walls_left,
        'p2_bfs_dist': p2_bfs_dist,
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


@app.route('/matchup')
def matchup_page():
    return render_template('matchup.html')

# ── API: matchup ──────────────────────────────────────────────────────

@app.route('/api/matchup/models')
def api_matchup_models():
    """Return list of available model files."""
    pts = sorted(glob.glob(os.path.join(MODEL_DIR, '*.pt')))
    saves_dir = MODEL_DIR + '_saves'
    if os.path.isdir(saves_dir):
        pts += sorted(glob.glob(os.path.join(saves_dir, '*.pt')))
    return jsonify([os.path.basename(p) for p in pts])


@app.route('/api/matchup/status')
def api_matchup_status():
    with _matchup_lock:
        snap = dict(_matchup_state)
        snap['games'] = []   # omit game data from status poll (use /api/matchup/games)
    return jsonify(snap)


@app.route('/api/matchup/games')
def api_matchup_games():
    with _matchup_lock:
        games = list(_matchup_state['games'])
    return jsonify(games)


@app.route('/api/matchup/start', methods=['POST'])
def api_matchup_start():
    global _matchup_cancel
    with _matchup_lock:
        if _matchup_state['status'] == 'running':
            return jsonify({'error': 'already running'}), 400

    body = request.get_json(force=True)

    def resolve_model(name):
        if os.path.isabs(name):
            return name
        # Try MODEL_DIR first, then saves dir
        for d in [MODEL_DIR, MODEL_DIR + '_saves']:
            p = os.path.join(d, name if name.endswith('.pt') else name + '.pt')
            if os.path.exists(p):
                return p
        return None

    path_a = resolve_model(body.get('model_a', 'best'))
    path_b = resolve_model(body.get('model_b', 'best'))
    if not path_a or not path_b:
        return jsonify({'error': f'model not found: {body.get("model_a")} / {body.get("model_b")}'}), 400

    games = max(2, int(body.get('games', 20)) // 2 * 2)  # round to even
    cfg = {
        'model_a': path_a, 'model_b': path_b,
        'games':   games,
        'pos_a':   float(body.get('pos_a',  _cfg.POSITION_PRIOR_BOOST)),
        'bfs_a':   float(body.get('bfs_a',  _cfg.BFS_MOVE_BOOST)),
        'sims_a':  int(body.get('sims_a', _cfg.PV_EVALUATE_COUNT)),
        'pos_b':   float(body.get('pos_b',  _cfg.POSITION_PRIOR_BOOST)),
        'bfs_b':   float(body.get('bfs_b',  _cfg.BFS_MOVE_BOOST)),
        'sims_b':  int(body.get('sims_b', _cfg.PV_EVALUATE_COUNT)),
    }

    _matchup_cancel = threading.Event()
    with _matchup_lock:
        _matchup_state.update({
            'status': 'running', 'cfg': cfg,
            'wins_a': 0, 'draws': 0, 'wins_b': 0,
            'completed': 0, 'total': games,
            'score_a': 0.0, 'score_b': 0.0,
            'games': [],
        })

    def _run():
        from evaluate_matchup import run_matchup

        def _on_game(g):
            with _matchup_lock:
                _matchup_state['games'].append(g)
                r = g['result']
                if r == 'a_win':   _matchup_state['wins_a'] += 1
                elif r == 'b_win': _matchup_state['wins_b'] += 1
                else:              _matchup_state['draws']  += 1
                c = _matchup_state['completed'] = (
                    _matchup_state['wins_a'] + _matchup_state['draws'] + _matchup_state['wins_b']
                )
                wa, d, wb = _matchup_state['wins_a'], _matchup_state['draws'], _matchup_state['wins_b']
                _matchup_state['score_a'] = round((wa + 0.5 * d) / c, 4)
                _matchup_state['score_b'] = round((wb + 0.5 * d) / c, 4)

        run_matchup(cfg, on_game=_on_game, cancel_flag=_matchup_cancel)
        with _matchup_lock:
            _matchup_state['status'] = 'cancelled' if _matchup_cancel.is_set() else 'done'
            # Persist the run so the Replay tab can browse it later
            snap = dict(_matchup_state)
        _save_matchup_replay(cfg, snap)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({'started': True, 'games': games})


@app.route('/api/matchup/cancel', methods=['POST'])
def api_matchup_cancel():
    with _matchup_lock:
        if _matchup_state['status'] != 'running':
            return jsonify({'error': 'not running'}), 400
    _matchup_cancel.set()
    return jsonify({'cancelled': True})


def _save_matchup_replay(cfg, snap):
    """Write completed matchup games to LOGS_DIR/matchup_games/<timestamp>.json."""
    from datetime import datetime
    mdir = os.path.join(LOGS_DIR, 'matchup_games')
    os.makedirs(mdir, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(mdir, f'{ts}.json')
    ma = os.path.basename(cfg['model_a'])
    mb = os.path.basename(cfg['model_b'])
    record = {
        'label':     f'{ma} vs {mb}',
        'model_a':   ma,
        'model_b':   mb,
        'sims_a':    cfg.get('sims_a'),
        'sims_b':    cfg.get('sims_b'),
        'score_a':   snap['score_a'],
        'score_b':   snap['score_b'],
        'wins_a':    snap['wins_a'],
        'draws':     snap['draws'],
        'wins_b':    snap['wins_b'],
        'cancelled': snap['status'] == 'cancelled',
        'games':     snap['games'],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f)


@app.route('/api/matchup_replays')
def api_matchup_replays():
    mdir = os.path.join(LOGS_DIR, 'matchup_games')
    if not os.path.exists(mdir):
        return jsonify([])
    result = []
    for fp in sorted(glob.glob(os.path.join(mdir, '*.json')), reverse=True):
        name = os.path.basename(fp)
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            result.append({
                'name':    name,
                'label':   meta.get('label', name),
                'score_a': meta.get('score_a'),
                'score_b': meta.get('score_b'),
                'games':   len(meta.get('games', [])),
            })
        except Exception:
            continue
    return jsonify(result)


@app.route('/api/matchup_replays/<name>')
def api_matchup_replay(name):
    mdir = os.path.join(LOGS_DIR, 'matchup_games')
    path = os.path.join(mdir, name)
    if not os.path.exists(path):
        return jsonify({'error': 'not found'}), 404
    with open(path, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))

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
    mcts_q_values = None
    if mcts_rollouts > 0 and model is not None:
        visit_probs, root_q, action_q = pv_mcts_full(model, deepcopy(state), mcts_rollouts)
        mcts_policies = {int(a): float(p) for a, p in zip(legal, visit_probs)}
        mcts_value    = root_q
        mcts_q_values = {int(a): q for a, q in zip(legal, action_q) if q is not None}

    # Scalar BFS distances for P1 and P2 — same logic as _state_to_display
    _p_dist, _e_dist = state.bfs_distances()
    fp = (depth % 2 == 0)
    p1_bfs_dist = int(_p_dist if fp else _e_dist)
    p2_bfs_dist = int(_e_dist if fp else _p_dist)

    return jsonify({
        'channels':      channels,
        'p_bfs':         to_grid(p_bfs_flat),
        'e_bfs':         to_grid(e_bfs_flat_abs),
        'legal_actions': legal,
        'policies':      policies,
        'value':         value,
        'mcts_policies': mcts_policies,
        'mcts_value':    mcts_value,
        'mcts_q_values': mcts_q_values,
        'p1_bfs_dist':   p1_bfs_dist,
        'p2_bfs_dist':   p2_bfs_dist,
        'N':             N,
        'n_channels':    n_channels,
        'use_bfs':       USE_BFS_CHANNELS,
    })



# ── API: training control ────────────────────────────────────────────────

@app.route('/api/training/status')
def api_training_status():
    running = _training_running()
    stop_pending = os.path.exists(_stop_flag_path())
    last_cycle = None
    stats_path = os.path.join(LOGS_DIR, 'stats.csv')
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if rows:
            last_cycle = int(rows[-1]['cycle'])
    # Last non-empty line from the most recent log file
    last_log_line = ''
    log_files = sorted(glob.glob(os.path.join(LOGS_DIR, '*.log')))
    if log_files:
        try:
            with open(log_files[-1], 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    stripped = line.rstrip()
                    if stripped:
                        last_log_line = stripped
        except OSError:
            pass
    return jsonify({
        'running': running,
        'stop_pending': stop_pending,
        'pid': _train_proc.pid if running else None,
        'last_cycle': last_cycle,
        'last_log_line': last_log_line,
    })


@app.route('/api/training/start', methods=['POST'])
def api_training_start():
    global _train_proc
    if _training_running():
        return jsonify({'error': 'already running', 'pid': _train_proc.pid}), 400
    # Clear any leftover stop flag
    sf = _stop_flag_path()
    if os.path.exists(sf):
        os.remove(sf)
    _train_proc = subprocess.Popen(
        [sys.executable, 'train_cycle.py'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return jsonify({'started': True, 'pid': _train_proc.pid})


@app.route('/api/training/stop', methods=['POST'])
def api_training_stop():
    """Graceful stop: write flag file; training exits after the current cycle."""
    if not _training_running():
        return jsonify({'error': 'not running'}), 400
    os.makedirs(LOGS_DIR, exist_ok=True)
    open(_stop_flag_path(), 'w').close()
    return jsonify({'stop_requested': True})


@app.route('/api/training/kill', methods=['POST'])
def api_training_kill():
    """Immediate termination of the training subprocess and all its children."""
    global _train_proc
    if not _training_running():
        return jsonify({'error': 'not running'}), 400
    pid = _train_proc.pid
    # On Windows, taskkill /F /T kills the entire process tree (including pool workers).
    # Plain terminate() only kills the top-level process, leaving pool workers as
    # orphans that then flood the logs with BrokenPipeError.
    import platform
    if platform.system() == 'Windows':
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            capture_output=True,
        )
    else:
        import signal
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    try:
        _train_proc.wait(timeout=5)
    except Exception:
        pass
    sf = _stop_flag_path()
    if os.path.exists(sf):
        os.remove(sf)
    return jsonify({'killed': True})


# ── API: live log SSE ────────────────────────────────────────────────

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
