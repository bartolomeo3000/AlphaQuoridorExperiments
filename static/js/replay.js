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

// ── Cycle / game list ──────────────────────────────────────────────────

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

function badgeHTML(result) {
  if (result === 'latest_win') return '<span class="badge badge-win">Latest ✓</span>';
  if (result === 'best_win')   return '<span class="badge badge-loss">Best ✓</span>';
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
}

function updateGameInfo() {
  if (!gameRecord) return;
  const res = gameRecord.result.replace(/_/g, ' ');
  const lf  = gameRecord.latest_first ? 'Latest = P1' : 'Latest = P2';
  document.getElementById('game-info').innerHTML =
    `Result: <strong>${res}</strong><br>${lf}<br>Moves: ${gameRecord.actions.length}`;
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

// ── Init ───────────────────────────────────────────────────────────────
loadCycles();
