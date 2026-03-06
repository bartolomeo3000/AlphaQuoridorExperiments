/* play.js — Human vs AI interactive game */

const N = 7;
const boardEl = document.getElementById('board');
const board   = new QuoridorBoard(boardEl, { N, cellSize: 62, wallSize: 8, gap: 4 });

let gameState   = null;
let humanSide   = 1;   // 1 = P1, 2 = P2, 0 = 2-player mode
let gameMode    = 'vs_ai';  // 'vs_ai' | 'vs_human'
let wallMode    = 'hwall';  // 'hwall' | 'vwall'
let gameActive  = false;
let canUndo     = false;

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

function onSideSelectChange() {
  // Rollout selector is always visible — used for AI moves and MCTS analysis
}

async function startNewGame() {
  humanSide        = parseInt(document.getElementById('side-select').value, 10);
  gameMode         = humanSide === 0 ? 'vs_human' : 'vs_ai';
  const rollouts   = parseInt(document.getElementById('rollout-select').value, 10);
  const human_first = gameMode === 'vs_human' || humanSide === 1;

  document.getElementById('page-title').textContent =
    gameMode === 'vs_human' ? '2 Player Game' : 'Human vs AI';

  if (gameMode !== 'vs_human') show('spinner-wrap');
  hide('setup-panel');
  hide('result-panel');

  const data = await fetch('/api/play/new', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rollouts, human_first, vs_human: gameMode === 'vs_human' }),
  }).then(r => r.json());

  hide('spinner-wrap');
  gameActive = true;
  canUndo    = false;
  gameState  = data;
  setWallMode('hwall');
  show('game-panel');
  updateUndoBtn();
  renderGame(data);
  fetchAndRenderAnalysis(data, getAnalysisRollouts());
}

// ── Board interaction ──────────────────────────────────────────────────

board.onClick(async (type, payload) => {
  if (!gameActive || !gameState) return;
  if (gameState.done) return;

  // Determine if it's the human's turn
  const humanTurn = gameMode === 'vs_human' ||
                    (gameState.is_p1_turn && humanSide === 1) ||
                    (!gameState.is_p1_turn && humanSide === 2);
  if (!humanTurn) return;

  let action = null;

  // When it's P2's turn the legal action list is in P2's flipped frame.
  // Convert absolute board coords → current-player frame before checking.
  const W   = (N - 1) * (N - 1);
  const isP2Turn = !gameState.is_p1_turn;
  const absToFrame = absPos => isP2Turn ? (N * N - 1 - absPos) : absPos;
  const wAbsToFrame = absWpos => isP2Turn ? (W - 1 - absWpos) : absWpos;

  if (type === 'cell') {
    const frameAction = absToFrame(payload.pos);
    if ((gameState.legal || []).includes(frameAction)) {
      action = frameAction;
    }
  } else if (type === 'hwall' && wallMode === 'hwall') {
    action = N * N + wAbsToFrame(payload.wpos);
    if (!(gameState.legal || []).includes(action)) action = null;
  } else if (type === 'vwall' && wallMode === 'vwall') {
    action = N * N + W + wAbsToFrame(payload.wpos);
    if (!(gameState.legal || []).includes(action)) action = null;
  }

  if (action === null) return;

  // Immediately render human's move locally so the board responds at once
  const effectiveSide = gameMode === 'vs_human' ? (gameState.is_p1_turn ? 1 : 2) : humanSide;
  const intermediate = applyHumanMoveLocally(gameState, action, effectiveSide);
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
  if (gameMode !== 'vs_human') show('spinner-wrap');
  gameActive = false;

  const data = await fetch('/api/play/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  }).then(r => r.json());

  hide('spinner-wrap');
  gameState = data;

  if (!data.error) {
    canUndo = data.can_undo ?? false;
    if (!data.done) gameActive = true;
    updateUndoBtn();
    renderGame(data);
    if (data.done) {
      showResult(data);
    }
    fetchAndRenderAnalysis(data, getAnalysisRollouts());
  } else {
    console.warn('Move error:', data.error);
    gameActive = true;
  }
}

function updateUndoBtn() {
  const btn = document.getElementById('btn-undo');
  if (!btn) return;
  btn.disabled = !canUndo || !gameActive || (gameState && gameState.done);
}

async function undoMove() {
  if (!canUndo || !gameActive) return;
  gameActive = false;
  updateUndoBtn();
  const data = await fetch('/api/play/undo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  }).then(r => r.json());
  if (!data.error) {
    gameState = data;
    canUndo = data.can_undo ?? false;
    renderGame(data);
    gameActive = true;
    fetchAndRenderAnalysis(data, getAnalysisRollouts());
  } else {
    console.warn('Undo error:', data.error);
    gameActive = true;
  }
  updateUndoBtn();
}

// ── Rendering ─────────────────────────────────────────────────────────

function renderGame(state) {
  // legal actions are in current-player frame; convert pawn moves to absolute for highlight
  const isP2Turn = !state.is_p1_turn;
  const legalAbs = (state.legal || []).map(a =>
    (a < N * N && isP2Turn) ? (N * N - 1 - a) : a
  );
  board.draw({ ...state, legal: legalAbs, highlight: new Set(legalAbs.filter(a => a < N * N)) });

  document.getElementById('p1-walls').textContent = state.p1_walls_left ?? '?';
  document.getElementById('p2-walls').textContent = state.p2_walls_left ?? '?';
  document.getElementById('depth-label').textContent = `Move ${state.depth}`;

  const ind = document.getElementById('turn-indicator');
  if (state.done) {
    ind.className = 'turn-indicator turn-done';
    ind.textContent = 'Game over';
  } else if (gameMode === 'vs_human') {
    ind.className = 'turn-indicator turn-human';
    ind.textContent = state.is_p1_turn ? '▶ P1\'s turn' : '▶ P2\'s turn';
  } else {
    const humanTurn = (state.is_p1_turn && humanSide === 1) ||
                      (!state.is_p1_turn && humanSide === 2);
    ind.className = humanTurn ? 'turn-indicator turn-human' : 'turn-indicator turn-ai';
    ind.textContent = humanTurn ? '▶ Your turn' : '⚙ AI thinking…';
  }
}

function showResult(state) {
  show('result-panel');
  hide('game-panel');
  const rtxt = document.getElementById('result-text');
  if (state.winner === 'draw') {
    rtxt.textContent = '½ Draw!';
    rtxt.style.color = 'var(--draw)';
  } else if (gameMode === 'vs_human') {
    const winner = state.winner === 'p1' ? 'P1' : 'P2';
    rtxt.textContent = `🎉 ${winner} wins!`;
    rtxt.style.color = 'var(--win)';
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
    fetchAndRenderAnalysis(gameState, getAnalysisRollouts());
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

async function fetchAndRenderAnalysis(state, mcts_rollouts = 0) {
  if (!analysisEnabled) return;
  const anSpinner = document.getElementById('an-spinner');
  anSpinner.classList.remove('hidden');

  const body = { ...displayToInternal(state), mcts_rollouts };
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

  // NN value chip
  const vProb = data.value != null ? (data.value + 1) / 2 : null;
  const vEl = document.getElementById('an-value');
  vEl.textContent = vProb != null ? (vProb * 100).toFixed(1) + '%' : '—';
  vEl.style.color = vProb != null
    ? (vProb > 0.55 ? 'var(--win)' : vProb < 0.45 ? 'var(--loss)' : 'var(--draw)') : '';

  // MCTS value chip
  const mctsChip = document.getElementById('an-mcts-value-chip');
  if (data.mcts_value != null) {
    const mv = (data.mcts_value + 1) / 2;
    const mvEl = document.getElementById('an-mcts-value');
    mvEl.textContent = (mv * 100).toFixed(1) + '%';
    mvEl.style.color = mv > 0.55 ? 'var(--win)' : mv < 0.45 ? 'var(--loss)' : 'var(--draw)';
    mctsChip.style.display = '';
  } else {
    mctsChip.style.display = 'none';
  }

  document.getElementById('an-p1bfs').textContent = state.p1_bfs_dist ?? '—';
  document.getElementById('an-p2bfs').textContent = state.p2_bfs_dist ?? '—';
  document.getElementById('an-legal').textContent = data.legal_actions?.length ?? '—';

  // Policy strip
  const hasMCTS = data.mcts_policies != null;
  document.getElementById('an-policy-title').textContent = hasMCTS
    ? 'Policy — top 15  \u25a0 NN prior  \u25a0 MCTS visits'
    : 'Policy — top 15 legal actions by prior';

  const strip = document.getElementById('an-policy');
  strip.innerHTML = '';
  if (data.policies) {
    const allEntries = Object.entries(data.policies)
      .map(([a, p]) => ({ a: parseInt(a), p }));
    const maxP = Math.max(...allEntries.map(e => e.p), 1e-9);
    const maxM = hasMCTS ? Math.max(...allEntries.map(e => data.mcts_policies[e.a] ?? 0), 1e-9) : 1;
    const entries = allEntries
      .sort((a, b) => hasMCTS
        ? (data.mcts_policies[b.a] ?? 0) - (data.mcts_policies[a.a] ?? 0)
        : b.p - a.p)
      .slice(0, 15);
    const BAR_H = 60;
    const isFP  = state.is_p1_turn;

    // Legend
    if (hasMCTS) {
      const legend = document.createElement('div');
      legend.style.cssText = 'display:flex;gap:12px;font-size:0.72rem;color:var(--text-dim);margin-bottom:6px;width:100%';
      legend.innerHTML =
        '<span><span style="display:inline-block;width:10px;height:10px;background:var(--accent);border-radius:2px;margin-right:3px"></span>NN prior</span>' +
        '<span><span style="display:inline-block;width:10px;height:10px;background:var(--accent2);border-radius:2px;margin-right:3px"></span>MCTS visits</span>';
      strip.appendChild(legend);
    }

    entries.forEach(({ a, p }, i) => {
      const hNN = Math.round((p / maxP) * BAR_H);
      const lbl = anActionLabel(a);
      const pct = (p * 100).toFixed(1);
      const div = document.createElement('div');
      div.className = 'policy-bar' + (i === 0 ? ' top-action' : '');

      if (hasMCTS) {
        const mp   = data.mcts_policies[a] ?? 0;
        const hM   = Math.round((mp / maxM) * BAR_H);
        const mpct = (mp * 100).toFixed(1);
        div.innerHTML = `
          <div class="bars-pair" style="height:${BAR_H}px">
            <div class="bar-nn"   style="height:${hNN}px" title="NN: ${pct}%"></div>
            <div class="bar-mcts" style="height:${hM}px"  title="MCTS: ${mpct}%"></div>
          </div>
          <div>${pct}%</div>
          <div style="color:var(--accent2)">${mpct}%</div>
          <div>${lbl}</div>`;
      } else {
        div.innerHTML = `
          <div style="height:${BAR_H}px;display:flex;align-items:flex-end">
            <div class="bar-inner" style="height:${hNN}px" title="${lbl}: ${pct}%"></div>
          </div>
          <div>${pct}%</div>
          <div>${lbl}</div>`;
      }

      div.addEventListener('mouseenter', () => board.setExternalHover(actionToHoverInfo(a, isFP)));
      div.addEventListener('mouseleave', () => board.clearExternalHover());
      div.addEventListener('click', () => {
        if (!gameActive || !gameState || gameState.done) return;
        const humanTurn = gameMode === 'vs_human' ||
                          (gameState.is_p1_turn && humanSide === 1) ||
                          (!gameState.is_p1_turn && humanSide === 2);
        if (!humanTurn) return;
        if (!(gameState.legal || []).includes(a)) return;
        board.clearExternalHover();
        const es = gameMode === 'vs_human' ? (gameState.is_p1_turn ? 1 : 2) : humanSide;
        const intermediate = applyHumanMoveLocally(gameState, a, es);
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

/** Run MCTS analysis on demand for the current game position. */
async function runMCTSAnalysis() {
  if (!gameState || !analysisEnabled) return;
  const rollouts = parseInt(document.getElementById('rollout-select').value, 10);
  await fetchAndRenderAnalysis(gameState, rollouts);
}

let autoMCTS = false;
function toggleAutoMCTS() {
  autoMCTS = !autoMCTS;
  const btn = document.getElementById('btn-mcts-auto');
  btn.textContent = autoMCTS ? 'Auto MCTS: On' : 'Auto MCTS: Off';
  btn.classList.toggle('btn-primary', autoMCTS);
}

function getAnalysisRollouts() {
  return autoMCTS ? parseInt(document.getElementById('rollout-select').value, 10) : 0;
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
