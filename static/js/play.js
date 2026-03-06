/* play.js — Human vs AI interactive game */

const N = 7;
const boardEl = document.getElementById('board');
const board   = new QuoridorBoard(boardEl, { N, cellSize: 62, wallSize: 8, gap: 4 });

let gameState   = null;
let humanSide   = 1;   // 1 = P1 (first player), 2 = P2
let wallMode    = 'hwall';  // 'hwall' | 'vwall'
let gameActive  = false;

// ── Wall orientation toggle ────────────────────────────────────────────

function setWallMode(mode) {
  wallMode = mode;
  board.onOverlayMode(mode);
  const btn = document.getElementById('wall-toggle');
  btn.textContent = mode === 'hwall' ? '━ Horizontal' : '┃ Vertical';
}

function toggleWallMode() {
  setWallMode(wallMode === 'hwall' ? 'vwall' : 'hwall');
}

// ── New game ────────────────────────────────────────────────────────────

document.getElementById('btn-new').addEventListener('click', startNewGame);
document.getElementById('btn-resign').addEventListener('click', startNewGame);
document.getElementById('btn-again').addEventListener('click', () => {
  show('setup-panel');
  hide('game-panel');
  hide('result-panel');
  hide('spinner-wrap');
  board.draw({ N, p1_pos: 45, p2_pos: 3, p1_walls_left: 5, p2_walls_left: 5, walls: new Array(36).fill(0), legal: [], done: false, winner: null });
});

async function startNewGame() {
  const rollouts   = parseInt(document.getElementById('rollout-select').value, 10);
  humanSide        = parseInt(document.getElementById('side-select').value, 10);
  const human_first = humanSide === 1;

  show('spinner-wrap');
  hide('setup-panel');
  hide('result-panel');

  const data = await fetch('/api/play/new', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rollouts, human_first }),
  }).then(r => r.json());

  hide('spinner-wrap');
  gameActive = true;
  gameState  = data;
  setWallMode('hwall');
  show('game-panel');
  renderGame(data);
  fetchAndRenderAnalysis(data);
}

// ── Board interaction ──────────────────────────────────────────────────

board.onClick(async (type, payload) => {
  if (!gameActive || !gameState) return;
  if (gameState.done) return;

  // Determine if it's the human's turn
  const humanTurn = (gameState.is_p1_turn && humanSide === 1) ||
                    (!gameState.is_p1_turn && humanSide === 2);
  if (!humanTurn) return;

  let action = null;

  if (type === 'cell') {
    // Cell click always = pawn move
    if ((gameState.legal || []).includes(payload.pos)) {
      action = payload.pos;
    }
  } else if (type === 'hwall' && wallMode === 'hwall') {
    action = N * N + payload.wpos;
    if (!(gameState.legal || []).includes(action)) action = null;
  } else if (type === 'vwall' && wallMode === 'vwall') {
    action = N * N + (N - 1) * (N - 1) + payload.wpos;
    if (!(gameState.legal || []).includes(action)) action = null;
  }

  if (action === null) return;

  // Immediately render human's move locally so the board responds at once
  const intermediate = applyHumanMoveLocally(gameState, action, humanSide);
  renderGame(intermediate);

  await sendMove(action);
});

/**
 * Apply the human's action to the display state locally (no server round-trip).
 * Used to give instant visual feedback before the AI responds.
 *
 * Wall indices:
 *   When it's P1's turn: legal wall wpos is in absolute (P1) frame — same as walls[].
 *   When it's P2's turn: legal wall wpos is in P2's current-player frame;
 *     absolute index = (N-1)²-1 - wpos  (mirrors State.rotate_walls reversal).
 */
function applyHumanMoveLocally(state, action, side) {
  const W = (N - 1) * (N - 1);
  const s = {
    ...state,
    walls: [...state.walls],
    legal: [],                        // clear legals — no longer valid
    is_p1_turn: !state.is_p1_turn,    // turn flips after the move
  };

  if (action < N * N) {
    // Pawn move — action is in current-player's frame
    if (side === 1) {
      s.p1_pos = action;              // P1 frame = absolute
    } else {
      s.p2_pos = N * N - 1 - action; // P2 frame is flipped; convert to absolute
    }
  } else if (action < N * N + W) {
    // Horizontal wall
    const wpos = action - N * N;
    const absWpos = side === 1 ? wpos : W - 1 - wpos;
    s.walls[absWpos] = 1;
    if (side === 1) s.p1_walls_left = (s.p1_walls_left || 1) - 1;
    else            s.p2_walls_left = (s.p2_walls_left || 1) - 1;
  } else {
    // Vertical wall
    const wpos = action - N * N - W;
    const absWpos = side === 1 ? wpos : W - 1 - wpos;
    s.walls[absWpos] = 2;
    if (side === 1) s.p1_walls_left = (s.p1_walls_left || 1) - 1;
    else            s.p2_walls_left = (s.p2_walls_left || 1) - 1;
  }

  return s;
}

async function sendMove(action) {
  show('spinner-wrap');
  gameActive = false;

  const data = await fetch('/api/play/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  }).then(r => r.json());

  hide('spinner-wrap');
  gameState = data;

  if (!data.error) {
    renderGame(data);
    if (data.done) {
      showResult(data);
    } else {
      gameActive = true;
    }
    fetchAndRenderAnalysis(data);
  } else {
    console.warn('Move error:', data.error);
    gameActive = true;
  }
}

// ── Rendering ─────────────────────────────────────────────────────────

function renderGame(state) {
  board.draw({ ...state, highlight: new Set(state.legal || []) });

  document.getElementById('p1-walls').textContent = state.p1_walls_left ?? '?';
  document.getElementById('p2-walls').textContent = state.p2_walls_left ?? '?';
  document.getElementById('depth-label').textContent = `Move ${state.depth}`;

  const humanTurn = (state.is_p1_turn && humanSide === 1) ||
                    (!state.is_p1_turn && humanSide === 2);
  const ind = document.getElementById('turn-indicator');
  if (state.done) {
    ind.className = 'turn-indicator turn-done';
    ind.textContent = 'Game over';
  } else if (humanTurn) {
    ind.className = 'turn-indicator turn-human';
    ind.textContent = '▶ Your turn';
  } else {
    ind.className = 'turn-indicator turn-ai';
    ind.textContent = '⚙ AI thinking…';
  }
}

function showResult(state) {
  show('result-panel');
  hide('game-panel');
  const rtxt = document.getElementById('result-text');
  if (state.winner === 'draw') {
    rtxt.textContent = '½ Draw!';
    rtxt.style.color = 'var(--draw)';
  } else {
    const humanWon = (state.winner === 'p1' && humanSide === 1) ||
                     (state.winner === 'p2' && humanSide === 2);
    rtxt.textContent  = humanWon ? '🎉 You win!' : '💀 AI wins!';
    rtxt.style.color  = humanWon ? 'var(--win)' : 'var(--loss)';
  }
}

// ── Utils ──────────────────────────────────────────────────────────────

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

// ── Analysis panel ────────────────────────────────────────────────────

let analysisEnabled = false;

function toggleAnalysis() {
  analysisEnabled = !analysisEnabled;
  const btn = document.getElementById('btn-analysis-toggle');
  btn.textContent = analysisEnabled ? '🔬 Hide Analysis' : '🔬 Show Analysis';
  btn.classList.toggle('btn-primary', analysisEnabled);
  const panel = document.getElementById('analysis-panel');
  panel.classList.toggle('hidden', !analysisEnabled);
  // If turned on mid-game, fetch immediately
  if (analysisEnabled && gameState && !gameState.done) {
    fetchAndRenderAnalysis(gameState);
  }
}

/** Convert display state (from server) back to internal State format for /api/inspect */
function displayToInternal(state) {
  const fp = state.is_p1_turn;
  let player, enemy, walls;
  if (fp) {
    player = [state.p1_pos, state.p1_walls_left];
    enemy  = [N * N - 1 - state.p2_pos, state.p2_walls_left];
    walls  = state.walls.slice();
  } else {
    player = [N * N - 1 - state.p2_pos, state.p2_walls_left];
    enemy  = [state.p1_pos, state.p1_walls_left];
    walls  = state.walls.slice().reverse();
  }
  return { player, enemy, walls, depth: state.depth };
}

async function fetchAndRenderAnalysis(state) {
  if (!analysisEnabled) return;
  const anSpinner = document.getElementById('an-spinner');
  anSpinner.classList.remove('hidden');

  const body = displayToInternal(state);
  let data;
  try {
    const resp = await fetch('/api/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    data = await resp.json();
  } catch (e) {
    console.warn('Analysis fetch failed', e);
    anSpinner.classList.add('hidden');
    return;
  }
  anSpinner.classList.add('hidden');

  // Summary stats
  const v     = data.value;
  const vEl   = document.getElementById('an-value');
  const vProb = v != null ? (v + 1) / 2 : null;  // tanh [-1,1] → win prob [0,1]
  vEl.textContent = vProb != null ? (vProb * 100).toFixed(1) + '%' : '—';
  vEl.style.color = vProb != null
    ? (vProb > 0.55 ? 'var(--win)' : vProb < 0.45 ? 'var(--loss)' : 'var(--draw)') : '';

  const p1r = Math.floor(state.p1_pos / N), p1c = state.p1_pos % N;
  const p2r = Math.floor(state.p2_pos / N), p2c = state.p2_pos % N;
  document.getElementById('an-p1bfs').textContent = data.p_bfs?.[p1r]?.[p1c] ?? '—';
  document.getElementById('an-p2bfs').textContent = data.e_bfs?.[p2r]?.[p2c] ?? '—';
  document.getElementById('an-legal').textContent = data.legal_actions?.length ?? '—';

  // Policy strip
  const strip = document.getElementById('an-policy');
  strip.innerHTML = '';
  if (data.policies) {
    const entries = Object.entries(data.policies)
      .map(([a, p]) => ({ a: parseInt(a), p }))
      .sort((b, a) => a.p - b.p)
      .slice(0, 15);
    const maxP = entries[0]?.p || 1;
    const BAR_H = 60;
    const isFP  = state.is_p1_turn;
    entries.forEach(({ a, p }, i) => {
      const h   = Math.round((p / maxP) * BAR_H);
      const lbl = anActionLabel(a);
      const pct = (p * 100).toFixed(1);
      const div = document.createElement('div');
      div.className = 'policy-bar' + (i === 0 ? ' top-action' : '');
      div.innerHTML = `
        <div style="height:${BAR_H}px;display:flex;align-items:flex-end">
          <div class="bar-inner" style="height:${h}px" title="${lbl}: ${pct}%"></div>
        </div>
        <div>${pct}%</div>
        <div>${lbl}</div>`;
      div.addEventListener('mouseenter', () => board.setExternalHover(actionToHoverInfo(a, isFP)));
      div.addEventListener('mouseleave', () => board.clearExternalHover());
      div.addEventListener('click', () => {
        if (!gameActive || !gameState || gameState.done) return;
        const humanTurn = (gameState.is_p1_turn && humanSide === 1) ||
                          (!gameState.is_p1_turn && humanSide === 2);
        if (!humanTurn) return;
        if (!(gameState.legal || []).includes(a)) return;
        board.clearExternalHover();
        const intermediate = applyHumanMoveLocally(gameState, a, humanSide);
        renderGame(intermediate);
        sendMove(a);
      });
      div.style.cursor = 'pointer';
      strip.appendChild(div);
    });
  } else {
    strip.textContent = 'No model loaded.';
  }

  // Heatmaps
  const CH_NAMES = [
    'Ch1: P1 Position', 'Ch2: P1 Walls Count',
    'Ch3: P2 Position (flipped)', 'Ch4: P2 Walls Count',
    'Ch5: H-edge Blocked', 'Ch6: V-edge Blocked',
    'Ch7: P1 BFS Distance', 'Ch8: P2 BFS Distance',
  ];
  const grid = document.getElementById('an-heatmaps');
  grid.innerHTML = '';
  (data.channels || []).forEach((values, i) => {
    const card = document.createElement('div');
    card.className = 'heatmap-card';
    const h4 = document.createElement('h4');
    h4.textContent = CH_NAMES[i] || `Ch${i + 1}`;
    card.appendChild(h4);
    const c = document.createElement('canvas');
    QuoridorBoard.drawHeatmap(c, values, { cellSize: 28 });
    card.appendChild(c);
    grid.appendChild(card);
  });
}

function anActionLabel(a) {
  const W = (N - 1) * (N - 1);
  if (a < N * N)     return `(${Math.floor(a / N)},${a % N})`;
  if (a < N * N + W) return `H${a - N * N}`;
  return `V${a - N * N - W}`;
}

/** Convert policy action (current-player frame) → board external-hover descriptor. */
function actionToHoverInfo(a, isFP) {
  const W = (N - 1) * (N - 1);
  if (a < N * N) {
    const absPos = isFP ? a : (N * N - 1 - a);
    return { type: 'cell', row: Math.floor(absPos / N), col: absPos % N };
  } else if (a < N * N + W) {
    const wpos    = a - N * N;
    const absWpos = isFP ? wpos : (W - 1 - wpos);
    return { type: 'hwall', wrow: Math.floor(absWpos / (N - 1)), wcol: absWpos % (N - 1) };
  } else {
    const wpos    = a - N * N - W;
    const absWpos = isFP ? wpos : (W - 1 - wpos);
    return { type: 'vwall', wrow: Math.floor(absWpos / (N - 1)), wcol: absWpos % (N - 1) };
  }
}

// Initial blank board draw
board.draw({
  N,
  p1_pos: N * (N - 1) + Math.floor(N / 2),
  p2_pos: Math.floor(N / 2),
  p1_walls_left: 5,
  p2_walls_left: 5,
  walls: new Array((N - 1) * (N - 1)).fill(0),
  legal: [],
  done: false,
  winner: null,
});
