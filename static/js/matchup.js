/* matchup.js — Model Matchup page */

const N  = 7;
const WS = 5;

const boardEl = document.getElementById('board');
const board   = new QuoridorBoard(boardEl, { N, cellSize: 62, wallSize: 8, gap: 4 });

let allStates   = [];
let stepIdx     = 0;
let playing     = false;
let playTimer   = null;
let curGame     = null;
let knownGames  = 0;   // how many games we've already rendered in the sidebar
let pollTimer   = null;
let lastCfg     = {};

// ── Model list ────────────────────────────────────────────────────────────────

async function loadModels() {
  const models = await fetch('/api/matchup/models').then(r => r.json());
  ['ma-model', 'mb-model'].forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    });
    // Default: best.pt for A, latest.pt for B
    const preferred = id === 'ma-model' ? 'best.pt' : 'latest.pt';
    const option = [...sel.options].find(o => o.value === preferred);
    if (option) sel.value = option.value;
  });
}

// ── Start / Cancel ────────────────────────────────────────────────────────────

async function startMatchup() {
  const cfg = {
    model_a: document.getElementById('ma-model').value,
    model_b: document.getElementById('mb-model').value,
    pos_a:   parseFloat(document.getElementById('ma-pos').value),
    bfs_a:   parseFloat(document.getElementById('ma-bfs').value),
    sims_a:  parseInt(document.getElementById('ma-sims').value, 10),
    pos_b:   parseFloat(document.getElementById('mb-pos').value),
    bfs_b:   parseFloat(document.getElementById('mb-bfs').value),
    sims_b:  parseInt(document.getElementById('mb-sims').value, 10),
    games:   parseInt(document.getElementById('m-games').value, 10),
  };
  lastCfg = cfg;

  const resp = await fetch('/api/matchup/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  });
  if (!resp.ok) {
    const err = await resp.json();
    alert('Error: ' + (err.error || resp.statusText));
    return;
  }

  // Reset UI
  knownGames = 0;
  document.getElementById('m-game-list').innerHTML = '';
  document.getElementById('score-row').style.display    = '';
  document.getElementById('replay-section').style.display = 'none';
  document.getElementById('label-a').textContent = cfg.model_a;
  document.getElementById('label-b').textContent = cfg.model_b;
  document.getElementById('btn-run').disabled    = true;
  document.getElementById('btn-cancel').disabled = false;
  allStates = [];
  curGame   = null;

  startPolling();
}

async function cancelMatchup() {
  await fetch('/api/matchup/cancel', { method: 'POST' });
}

// ── Polling ───────────────────────────────────────────────────────────────────

function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(pollStatus, 1500);
  pollStatus();
}

async function pollStatus() {
  const st = await fetch('/api/matchup/status').then(r => r.json());
  updateScoreRow(st);

  if (st.completed > knownGames) {
    const games = await fetch('/api/matchup/games').then(r => r.json());
    refreshGameList(games, st);
    if (knownGames === 0 && games.length > 0) {
      selectGame(games, 0);
    }
    knownGames = games.length;
  }

  if (st.status === 'done' || st.status === 'cancelled') {
    clearInterval(pollTimer);
    document.getElementById('btn-run').disabled    = false;
    document.getElementById('btn-cancel').disabled = true;
    const txt = st.status === 'done'
      ? `Done — ${st.completed} games`
      : `Cancelled after ${st.completed} games`;
    document.getElementById('m-status-text').textContent = txt;
  }
}

// ── Score row ─────────────────────────────────────────────────────────────────

function updateScoreRow(st) {
  const c = st.completed || 0;
  const t = st.total     || 1;
  document.getElementById('val-a').textContent    = c ? st.score_a.toFixed(3) : '—';
  document.getElementById('val-b').textContent    = c ? st.score_b.toFixed(3) : '—';
  document.getElementById('record-a').textContent = `${st.wins_a}W / ${st.draws}D / ${st.wins_b}L`;
  document.getElementById('record-b').textContent = `${st.wins_b}W / ${st.draws}D / ${st.wins_a}L`;
  document.getElementById('prog-label').textContent = `${c} / ${t}`;
  document.getElementById('prog-bar').style.width =
    (c / t * 100).toFixed(1) + '%';

  const statusMap = { running: 'Running…', done: 'Done', cancelled: 'Cancelled', idle: '' };
  document.getElementById('m-status-text').textContent = statusMap[st.status] || '';
}

// ── Game list ─────────────────────────────────────────────────────────────────

function badgeHTML(result) {
  if (result === 'a_win') return '<span class="badge badge-win">A ✓</span>';
  if (result === 'b_win') return '<span class="badge badge-loss">B ✓</span>';
  return '<span class="badge badge-draw">Draw</span>';
}

function refreshGameList(games, st) {
  const ul = document.getElementById('m-game-list');
  // Append only the new entries
  for (let i = knownGames; i < games.length; i++) {
    const g  = games[i];
    const li = document.createElement('li');
    li.innerHTML = `<span>#${i + 1}</span>${badgeHTML(g.result)}`
      + `<span class="text-dim" style="font-size:0.75rem">${g.actions.length} moves</span>`;
    li.dataset.idx = i;
    li.addEventListener('click', () => {
      fetch('/api/matchup/games').then(r => r.json()).then(all => selectGame(all, i));
    });
    ul.appendChild(li);
  }
  if (games.length > 0) {
    document.getElementById('replay-section').style.display = '';
  }
}

// ── Replay ────────────────────────────────────────────────────────────────────

function selectGame(games, idx) {
  document.querySelectorAll('#m-game-list li').forEach((li, i) =>
    li.classList.toggle('active', i === idx));

  curGame   = games[idx];
  allStates = GameState.replayActions(curGame.actions, N, WS);
  stepIdx   = 0;
  stopPlay();
  renderStep();
  updateGameInfo();
}

function renderStep() {
  if (!allStates.length) return;
  const dispState = { ...allStates[stepIdx] };
  if (stepIdx > 0 && curGame) {
    const lastAction = curGame.actions[stepIdx - 1];
    dispState.lastMove = lastAction < N * N ? lastAction : null;
  }
  board.draw(dispState);
  document.getElementById('step-label').textContent =
    `${stepIdx} / ${allStates.length - 1}`;

  let moveText = stepIdx === 0 ? 'Start' : '';
  if (stepIdx > 0 && curGame) {
    const a       = curGame.actions[stepIdx - 1];
    const turnNum = stepIdx;
    const mover   = (turnNum % 2 === 1) ? 'P1 (A)' : 'P2 (B)';
    const whoFirst = curGame.a_first ? '  ·  A=P1' : '  ·  A=P2';
    if (a < N * N) {
      const r = Math.floor(a / N), c = a % N;
      moveText = `Move ${turnNum}: ${mover} → (${r},${c})${whoFirst}`;
    } else if (a < N * N + (N - 1) ** 2) {
      const wp = a - N * N;
      moveText = `Move ${turnNum}: ${mover} H-wall @${wp}${whoFirst}`;
    } else {
      const wp = a - N * N - (N - 1) ** 2;
      moveText = `Move ${turnNum}: ${mover} V-wall @${wp}${whoFirst}`;
    }
  }
  document.getElementById('move-info').textContent = moveText;
}

function updateGameInfo() {
  if (!curGame) return;
  const resMap = { a_win: 'Model A wins', b_win: 'Model B wins', draw: 'Draw' };
  const first  = curGame.a_first ? 'A = P1' : 'A = P2';
  document.getElementById('game-info').innerHTML =
    `Result: <strong>${resMap[curGame.result] || curGame.result}</strong>`
    + `<br>${first}<br>Moves: ${curGame.actions.length}`;
}

// ── Playback controls ─────────────────────────────────────────────────────────

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
    if (stepIdx < allStates.length - 1) scheduleNext();
    else stopPlay();
  }, delay);
}

document.getElementById('btn-prev').addEventListener('click', () => { stopPlay(); stepTo(stepIdx - 1); });
document.getElementById('btn-next').addEventListener('click', () => { stopPlay(); stepTo(stepIdx + 1); });
document.getElementById('btn-play').addEventListener('click', () => {
  if (playing) stopPlay(); else startPlay();
});

// ── Init ──────────────────────────────────────────────────────────────────────

const basename = p => (p || '').split(/[\\/]/).pop() || p;

loadModels();

// Resume polling if a matchup is already running (page reload)
fetch('/api/matchup/status').then(r => r.json()).then(st => {
  if (st.status === 'running') {
    document.getElementById('score-row').style.display = '';
    document.getElementById('btn-run').disabled    = true;
    document.getElementById('btn-cancel').disabled = false;
    if (st.cfg) {
      document.getElementById('label-a').textContent = basename(st.cfg.model_a) || 'Model A';
      document.getElementById('label-b').textContent = basename(st.cfg.model_b) || 'Model B';
    }
    updateScoreRow(st);
    startPolling();
  }
});
