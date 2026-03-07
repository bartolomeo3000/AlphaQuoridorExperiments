/* replay.js — Eval game replay viewer */

const N   = 7;
const WS  = 5;  // walls per player

const boardEl = document.getElementById('board');
const board   = new QuoridorBoard(boardEl, { N, cellSize: 62, wallSize: 8, gap: 4 });

let allStates  = [];   // array of display dicts for current game
let stepIdx    = 0;
let playing    = false;
let playTimer  = null;
let gameRecord = null; // raw JSON record for current game
let curSource  = 'eval'; // 'eval' | 'matchup'

// ── Source toggle ────────────────────────────────────────────────────────

function setSource(src) {
  curSource = src;
  document.getElementById('panel-eval').style.display    = src === 'eval'    ? '' : 'none';
  document.getElementById('panel-matchup').style.display = src === 'matchup' ? '' : 'none';
  document.getElementById('src-eval').classList.toggle('active', src === 'eval');
  document.getElementById('src-matchup').classList.toggle('active', src === 'matchup');
  document.getElementById('game-list').innerHTML = '';
  gameRecord = null;
}

// ── Eval cycle source ──────────────────────────────────────────────────

async function loadCycles() {
  const list = await fetch('/api/eval_games').then(r => r.json());
  const sel  = document.getElementById('cycle-select');
  list.forEach(({ cycle }) => {
    const opt = document.createElement('option');
    opt.value = cycle;
    opt.textContent = `Cycle ${cycle}`;
    sel.appendChild(opt);
  });
  if (list.length) {
    sel.value = list[list.length - 1].cycle;
    await loadGames(list[list.length - 1].cycle);
  }
}

async function loadGames(cycle) {
  const data = await fetch(`/api/eval_games/${cycle}`).then(r => r.json());
  const ul   = document.getElementById('game-list');
  ul.innerHTML = '';
  (data.games || []).forEach((g, i) => {
    const li = document.createElement('li');
    const badge = badgeHTML(g.result);
    li.innerHTML = `<span>#${i + 1}</span>${badge}<span class="text-dim" style="font-size:0.75rem">${g.actions.length} moves</span>`;
    li.dataset.idx = i;
    li.addEventListener('click', () => selectGame(data.games, i));
    ul.appendChild(li);
  });
  // Auto-select first game
  if (data.games && data.games.length) selectGame(data.games, 0);
}

// ── Matchup source ─────────────────────────────────────────────────────

async function loadMatchupRuns() {
  const list = await fetch('/api/matchup_replays').then(r => r.json());
  const sel  = document.getElementById('matchup-select');
  sel.innerHTML = '<option value="">— pick run —</option>';
  list.forEach(run => {
    const opt = document.createElement('option');
    opt.value = run.name;
    opt.textContent = run.label + ` (${run.games}g)`;
    sel.appendChild(opt);
  });
  if (list.length) {
    sel.value = list[0].name;
    await loadMatchupGames(list[0]);
  }
}

async function loadMatchupGames(runMeta) {
  const data = await fetch(`/api/matchup_replays/${runMeta.name}`).then(r => r.json());
  // Show meta
  const metaEl = document.getElementById('matchup-meta');
  const sa = data.score_a != null ? data.score_a.toFixed(3) : '?';
  const sb = data.score_b != null ? data.score_b.toFixed(3) : '?';
  metaEl.textContent = `A: ${data.model_a}  ${sa} – ${sb}  B: ${data.model_b}`;

  const ul = document.getElementById('game-list');
  ul.innerHTML = '';
  (data.games || []).forEach((g, i) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>#${i + 1}</span>${badgeHTML(g.result)}`
      + `<span class="text-dim" style="font-size:0.75rem">${g.actions.length} moves</span>`;
    li.dataset.idx = i;
    li.addEventListener('click', () => selectGame(data.games, i));
    ul.appendChild(li);
  });
  if (data.games && data.games.length) selectGame(data.games, 0);
}

function badgeHTML(result) {
  if (result === 'latest_win' || result === 'a_win') return '<span class="badge badge-win">A / Latest ✓</span>';
  if (result === 'best_win'   || result === 'b_win') return '<span class="badge badge-loss">B / Best ✓</span>';
  return '<span class="badge badge-draw">Draw</span>';
}

function selectGame(games, idx) {
  // Highlight in sidebar
  document.querySelectorAll('#game-list li').forEach((li, i) =>
    li.classList.toggle('active', i === idx));

  gameRecord = games[idx];
  allStates  = GameState.replayActions(gameRecord.actions, N, WS);
  stepIdx    = 0;
  stopPlay();
  renderStep();
  updateGameInfo();
}

// ── Rendering ─────────────────────────────────────────────────────────

function renderStep() {
  if (!allStates.length) return;
  const state = allStates[stepIdx];

  // Annotate last move
  const dispState = { ...state };
  if (stepIdx > 0 && gameRecord) {
    const lastAction = gameRecord.actions[stepIdx - 1];
    dispState.lastMove = lastAction < N * N ? lastAction : null;
  }

  board.draw(dispState);
  document.getElementById('step-label').textContent =
    `${stepIdx} / ${allStates.length - 1}`;

  // Move info
  let moveText = stepIdx === 0 ? 'Start' : '';
  if (stepIdx > 0 && gameRecord) {
    const a = gameRecord.actions[stepIdx - 1];
    const turnNum = stepIdx;
    const mover   = (turnNum % 2 === 1) ? 'P1' : 'P2';
    if (a < N * N) {
      const r = Math.floor(a / N), c = a % N;
      moveText = `Move ${turnNum}: ${mover} → (${r},${c})`;
    } else if (a < N * N + (N-1)**2) {
      const wp = a - N * N;
      moveText = `Move ${turnNum}: ${mover} H-wall @${wp}`;
    } else {
      const wp = a - N * N - (N-1)**2;
      moveText = `Move ${turnNum}: ${mover} V-wall @${wp}`;
    }
  }
  document.getElementById('move-info').textContent = moveText;

  // Refresh analysis panel if open
  if (analysisEnabled && allStates.length) fetchAndRenderAnalysis(allStates[stepIdx]);
}

function updateGameInfo() {
  if (!gameRecord) return;
  // Support both eval format (latest_first) and matchup format (a_first)
  let firstLine = '';
  if (gameRecord.latest_first != null)
    firstLine = gameRecord.latest_first ? 'Latest = P1' : 'Latest = P2';
  else if (gameRecord.a_first != null)
    firstLine = gameRecord.a_first ? 'A = P1' : 'A = P2';
  const resMap = {
    latest_win: 'Latest wins', best_win: 'Best wins',
    a_win: 'A wins', b_win: 'B wins', draw: 'Draw',
  };
  const res = resMap[gameRecord.result] || gameRecord.result.replace(/_/g, ' ');
  document.getElementById('game-info').innerHTML =
    `Result: <strong>${res}</strong><br>${firstLine}<br>Moves: ${gameRecord.actions.length}`;
}

// ── Controls ───────────────────────────────────────────────────────────

function stepTo(n) {
  if (!allStates.length) return;
  stepIdx = Math.max(0, Math.min(allStates.length - 1, n));
  renderStep();
}

function startPlay() {
  if (!allStates.length || stepIdx >= allStates.length - 1) return;
  playing = true;
  document.getElementById('btn-play').textContent = '⏸ Pause';
  scheduleNext();
}

function stopPlay() {
  playing = false;
  clearTimeout(playTimer);
  document.getElementById('btn-play').textContent = '▶ Play';
}

function scheduleNext() {
  if (!playing) return;
  const speed = parseInt(document.getElementById('speed-slider').value, 10);
  const delay = Math.max(80, 1200 - speed * 110);
  playTimer = setTimeout(() => {
    stepTo(stepIdx + 1);
    if (stepIdx < allStates.length - 1) {
      scheduleNext();
    } else {
      stopPlay();
    }
  }, delay);
}

document.getElementById('btn-prev').addEventListener('click', () => { stopPlay(); stepTo(stepIdx - 1); });
document.getElementById('btn-next').addEventListener('click', () => { stopPlay(); stepTo(stepIdx + 1); });
document.getElementById('btn-play').addEventListener('click', () => {
  if (playing) stopPlay(); else startPlay();
});

document.getElementById('cycle-select').addEventListener('change', e => {
  if (e.target.value) loadGames(Number(e.target.value));
});

document.getElementById('matchup-select').addEventListener('change', e => {
  if (e.target.value) {
    const opt = [...document.getElementById('matchup-select').options]
      .find(o => o.value === e.target.value);
    loadMatchupGames({ name: e.target.value });
  }
});

// ── Analysis panel ────────────────────────────────────────────────────

let analysisEnabled = false;
let autoMCTS = false;

function toggleAnalysis() {
  analysisEnabled = !analysisEnabled;
  const btn = document.getElementById('btn-analysis-toggle');
  btn.textContent = analysisEnabled ? '🔬 Hide Analysis' : '🔬 Show Analysis';
  btn.classList.toggle('btn-primary', analysisEnabled);
  document.getElementById('analysis-panel').classList.toggle('hidden', !analysisEnabled);
  document.getElementById('rollout-wrap-replay').style.display = analysisEnabled ? '' : 'none';
  if (analysisEnabled && allStates.length) fetchAndRenderAnalysis(allStates[stepIdx]);
}

function toggleAutoMCTS() {
  autoMCTS = !autoMCTS;
  const btn = document.getElementById('btn-mcts-auto');
  btn.textContent = autoMCTS ? 'Auto MCTS: On' : 'Auto MCTS: Off';
  btn.classList.toggle('btn-primary', autoMCTS);
}

function getAnalysisRollouts() {
  return autoMCTS ? parseInt(document.getElementById('rollout-select').value, 10) : 0;
}

async function runMCTSAnalysis() {
  if (!allStates.length || !analysisEnabled) return;
  const rollouts = parseInt(document.getElementById('rollout-select').value, 10);
  await fetchAndRenderAnalysis(allStates[stepIdx], rollouts);
}

/** Convert display state → internal State format for /api/inspect */
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

function anActionLabel(a) {
  const W = (N - 1) * (N - 1);
  if (a < N * N)     return `(${Math.floor(a / N)},${a % N})`;
  if (a < N * N + W) return `H${a - N * N}`;
  return `V${a - N * N - W}`;
}

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

async function fetchAndRenderAnalysis(state, mcts_rollouts = 0) {
  if (!analysisEnabled) return;

  // Debounce: if a newer call supersedes this one, abort
  fetchAndRenderAnalysis._seq = (fetchAndRenderAnalysis._seq || 0) + 1;
  const mySeq = fetchAndRenderAnalysis._seq;

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

  // Discard stale response if user has moved on
  if (fetchAndRenderAnalysis._seq !== mySeq) return;
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

  document.getElementById('an-p1bfs').textContent = data.p1_bfs_dist ?? '—';
  document.getElementById('an-p2bfs').textContent = data.p2_bfs_dist ?? '—';
  document.getElementById('an-legal').textContent = data.legal_actions?.length ?? '—';

  // Policy strip
  const hasMCTS = data.mcts_policies != null;
  document.getElementById('an-policy-title').textContent = hasMCTS
    ? 'Policy — top 15  ■ NN prior  ■ MCTS visits  ■ MCTS Q'
    : 'Policy — top 15 legal actions by prior';

  const strip = document.getElementById('an-policy');
  strip.innerHTML = '';
  if (data.policies) {
    const allEntries = Object.entries(data.policies).map(([a, p]) => ({ a: parseInt(a), p }));
    const maxP = Math.max(...allEntries.map(e => e.p), 1e-9);
    const maxM = hasMCTS ? Math.max(...allEntries.map(e => data.mcts_policies[e.a] ?? 0), 1e-9) : 1;
    const entries = allEntries
      .sort((a, b) => hasMCTS
        ? (data.mcts_policies[b.a] ?? 0) - (data.mcts_policies[a.a] ?? 0)
        : b.p - a.p)
      .slice(0, 15);
    const BAR_H = 60;
    const isFP  = state.is_p1_turn;

    if (hasMCTS) {
      const legend = document.createElement('div');
      legend.style.cssText = 'display:flex;gap:12px;font-size:0.72rem;color:var(--text-dim);margin-bottom:6px;width:100%';
      legend.innerHTML =
        '<span><span style="display:inline-block;width:10px;height:10px;background:var(--accent);border-radius:2px;margin-right:3px"></span>NN prior</span>' +
        '<span><span style="display:inline-block;width:10px;height:10px;background:var(--accent2);border-radius:2px;margin-right:3px"></span>MCTS visits</span>' +
        '<span><span style="display:inline-block;width:10px;height:10px;background:#f5a623;border-radius:2px;margin-right:3px"></span>MCTS Q (win%)</span>';
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
        const qRaw  = data.mcts_q_values?.[a];
        const qProb = qRaw != null ? (qRaw + 1) / 2 : null;
        const hQ    = qProb != null ? Math.round(qProb * BAR_H) : 2;
        const qpct  = qProb != null ? (qProb * 100).toFixed(1) + '%' : '\u2014';
        div.innerHTML = `
          <div class="bars-pair" style="height:${BAR_H}px">
            <div class="bar-nn"   style="height:${hNN}px" title="NN: ${pct}%"></div>
            <div class="bar-mcts" style="height:${hM}px"  title="MCTS: ${mpct}%"></div>
            <div class="bar-q"    style="height:${hQ}px;opacity:${qProb != null ? 1 : 0.25}" title="Q: ${qpct}"></div>
          </div>
          <div>${pct}%</div>
          <div style="color:var(--accent2)">${mpct}%</div>
          <div style="color:#f5a623">${qpct}</div>
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
      strip.appendChild(div);
    });
  } else {
    strip.textContent = 'No model loaded.';
  }

  // Input channel heatmaps
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

// ── Init ───────────────────────────────────────────────────────────────
loadCycles();
loadMatchupRuns();
