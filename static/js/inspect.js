/* inspect.js — Position inspector editor + analysis */

const N = 7;
const boardEl = document.getElementById('board');
const board   = new QuoridorBoard(boardEl, { N, cellSize: 62, wallSize: 8, gap: 4 });

// ── Editor state ───────────────────────────────────────────────────────

let editorState = {
  p1_pos: N * (N - 1) + Math.floor(N / 2),  // player[0]
  p2_pos: Math.floor(N / 2),                  // enemy[0]
  walls: new Array((N - 1) * (N - 1)).fill(0),
};

let currentTool = 'move-p1';

// Map from API "display" convention to internal State convention
function displayToInternal() {
  const p1w = parseInt(document.getElementById('p1-walls-in').value, 10) || 0;
  const p2w = parseInt(document.getElementById('p2-walls-in').value, 10) || 0;
  const depth = parseInt(document.getElementById('depth-in').value, 10) || 0;
  const fp = depth % 2 === 0;

  // In the Python State, player[0] is in CURRENT PLAYER's frame.
  // Frame: current player starts at row N-1.
  // P1's absolute pos == internal player[0] when depth is even.
  // When depth is odd, P2 is the current player, their pos in their frame
  // is (N²-1 - p2_abs).
  let player, enemy, walls;
  if (fp) {
    // P1 is the current player → player slot = P1, enemy slot = P2 (needs flip)
    player = [editorState.p1_pos, p1w];
    enemy  = [N * N - 1 - editorState.p2_pos, p2w];
    walls  = editorState.walls.slice();
  } else {
    // P2 is the current player → player slot = P2 (flipped), enemy = P1
    player = [N * N - 1 - editorState.p2_pos, p2w];
    enemy  = [editorState.p1_pos, p1w];
    walls  = editorState.walls.slice().reverse();
  }
  return { player, enemy, walls, depth };
}

// ── Tools ─────────────────────────────────────────────────────────────

function setTool(tool) {
  currentTool = tool;
  document.querySelectorAll('.tool-btn').forEach(b =>
    b.classList.toggle('active', b.id === 'tool-' + tool));
  if (tool === 'hwall') board.onOverlayMode('hwall');
  else if (tool === 'vwall') board.onOverlayMode('vwall');
  else board.onOverlayMode(null);
}

board.onClick((type, payload) => {
  if (type === 'cell') {
    if (currentTool === 'move-p1') {
      editorState.p1_pos = payload.pos;
    } else if (currentTool === 'move-p2') {
      editorState.p2_pos = payload.pos;
    }
  } else if (type === 'hwall') {
    if (currentTool === 'hwall') {
      editorState.walls[payload.wpos] = 1;
    } else if (currentTool === 'rmwall') {
      editorState.walls[payload.wpos] = 0;
    }
  } else if (type === 'vwall') {
    if (currentTool === 'vwall') {
      editorState.walls[payload.wpos] = 2;
    } else if (currentTool === 'rmwall') {
      editorState.walls[payload.wpos] = 0;
    }
  }
  redrawBoard();
});

function redrawBoard() {
  board.draw({
    N,
    p1_pos:        editorState.p1_pos,
    p2_pos:        editorState.p2_pos,
    p1_walls_left: parseInt(document.getElementById('p1-walls-in').value, 10) || 0,
    p2_walls_left: parseInt(document.getElementById('p2-walls-in').value, 10) || 0,
    walls:         editorState.walls,
    legal: [],
    done: false,
    winner: null,
  });
}

function resetBoard() {
  editorState = {
    p1_pos: N * (N - 1) + Math.floor(N / 2),
    p2_pos: Math.floor(N / 2),
    walls: new Array((N - 1) * (N - 1)).fill(0),
  };
  document.getElementById('p1-walls-in').value = 5;
  document.getElementById('p2-walls-in').value = 5;
  document.getElementById('depth-in').value    = 0;
  document.getElementById('results').classList.add('hidden');
  redrawBoard();
}

function copyState() {
  const st = displayToInternal();
  navigator.clipboard.writeText(JSON.stringify(st, null, 2))
    .then(() => alert('Copied!'))
    .catch(() => prompt('Copy this:', JSON.stringify(st)));
}

// ── Analysis ───────────────────────────────────────────────────────────

async function analyze() {
  const st = displayToInternal();
  const resp = await fetch('/api/inspect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(st),
  });
  const data = await resp.json();

  document.getElementById('results').classList.remove('hidden');

  // Summary stats
  const v     = data.value;
  const vEl   = document.getElementById('value-out');
  const vProb = v != null ? (v + 1) / 2 : null;  // tanh [-1,1] → win prob [0,1]
  vEl.textContent = vProb != null ? (vProb * 100).toFixed(1) + '%' : '—';
  vEl.style.color = vProb != null
    ? (vProb > 0.55 ? 'var(--win)' : vProb < 0.45 ? 'var(--loss)' : 'var(--draw)')
    : '';

  // BFS distances for current player's position
  const p1r = Math.floor(editorState.p1_pos / N);
  const p1c = editorState.p1_pos % N;
  const p2r = Math.floor(editorState.p2_pos / N);
  const p2c = editorState.p2_pos % N;
  document.getElementById('p1-bfs-out').textContent = data.p_bfs?.[p1r]?.[p1c] ?? '—';
  document.getElementById('p2-bfs-out').textContent = data.e_bfs?.[p2r]?.[p2c] ?? '—';
  document.getElementById('legal-count-out').textContent = data.legal_actions?.length ?? '—';

  // Policy strip
  renderPolicyStrip(data);

  // Heatmaps
  renderHeatmaps(data);

  // Highlight legal moves on board
  board.draw({
    N,
    p1_pos:        editorState.p1_pos,
    p2_pos:        editorState.p2_pos,
    p1_walls_left: parseInt(document.getElementById('p1-walls-in').value, 10) || 0,
    p2_walls_left: parseInt(document.getElementById('p2-walls-in').value, 10) || 0,
    walls:         editorState.walls,
    legal:         data.legal_actions || [],
    highlight:     new Set(data.legal_actions?.filter(a => a < N * N) || []),
    done: false,
    winner: null,
  });
}

function renderPolicyStrip(data) {
  const strip = document.getElementById('policy-strip');
  strip.innerHTML = '';
  if (!data.policies) { strip.textContent = 'No model loaded.'; return; }

  // Sort by prob descending, take top 15
  const entries = Object.entries(data.policies)
    .map(([a, p]) => ({ a: parseInt(a), p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 15);

  const maxP = entries[0]?.p || 1;
  const BAR_HEIGHT = 60;

  entries.forEach(({ a, p }, i) => {
    const div = document.createElement('div');
    div.className = 'policy-bar' + (i === 0 ? ' top-action' : '');

    const h = Math.round((p / maxP) * BAR_HEIGHT);
    const label = actionLabel(a);
    const pct   = (p * 100).toFixed(1);

    div.innerHTML = `
      <div style="height:${BAR_HEIGHT}px;display:flex;align-items:flex-end">
        <div class="bar-inner" style="height:${h}px" title="${label}: ${pct}%"></div>
      </div>
      <div>${pct}%</div>
      <div>${label}</div>
    `;

    const isFP = (parseInt(document.getElementById('depth-in').value) || 0) % 2 === 0;
    div.addEventListener('mouseenter', () => board.setExternalHover(actionToHoverInfo(a, isFP)));
    div.addEventListener('mouseleave', () => board.clearExternalHover());

    strip.appendChild(div);
  });
}

function actionLabel(a) {
  if (a < N * N) {
    const r = Math.floor(a / N), c = a % N;
    return `(${r},${c})`;
  } else if (a < N * N + (N - 1) * (N - 1)) {
    return `H${a - N * N}`;
  } else {
    return `V${a - N * N - (N - 1) * (N - 1)}`;
  }
}

/**
 * Convert a policy action index (in current-player frame) to a board.js
 * external-hover descriptor (absolute / P1-frame coordinates).
 */
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

function renderHeatmaps(data) {
  const grid  = document.getElementById('heatmap-grid');
  grid.innerHTML = '';

  const CH_NAMES = [
    'Ch1: P1 Position',
    'Ch2: P1 Walls Count',
    'Ch3: P2 Position (flipped)',
    'Ch4: P2 Walls Count',
    'Ch5: H-edge Blocked',
    'Ch6: V-edge Blocked',
    'Ch7: P1 BFS Distance',
    'Ch8: P2 BFS Distance',
  ];

  const allGrids = (data.channels || []).map((grid, i) => ({ label: CH_NAMES[i] || `Ch${i+1}`, values: grid }));

  allGrids.forEach(({ label, values }) => {
    if (!values) return;
    const card = document.createElement('div');
    card.className = 'heatmap-card';

    const h4 = document.createElement('h4');
    h4.textContent = label;
    card.appendChild(h4);

    const c = document.createElement('canvas');
    QuoridorBoard.drawHeatmap(c, values, { cellSize: 28 });
    card.appendChild(c);

    grid.appendChild(card);
  });
}

// ── Init ───────────────────────────────────────────────────────────────
redrawBoard();
