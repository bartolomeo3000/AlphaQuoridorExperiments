/* dashboard.js — Training Dashboard logic */

const CHART_DEFAULTS = {
  responsive: true,
  animation: false,
  plugins: { legend: { labels: { color: '#8890ab', boxWidth: 12, font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: '#8890ab', font: { size: 10 } }, grid: { color: '#2e3248' } },
    y: { ticks: { color: '#8890ab', font: { size: 10 } }, grid: { color: '#2e3248' } },
  },
};

function mkChart(id, type, datasets, opts = {}) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  return new Chart(canvas, {
    type,
    data: { labels: [], datasets },
    options: { ...CHART_DEFAULTS, ...opts },
  });
}

function color(hex, alpha = 1) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Chart instances ────────────────────────────────────────────────────

const charts = {};

function initCharts() {
  charts.outcomes = mkChart('c-outcomes', 'bar', [
    { label: 'Win %',  data: [], backgroundColor: color('#4caf7d', 0.8), stack: 's' },
    { label: 'Draw %', data: [], backgroundColor: color('#e0a84a', 0.8), stack: 's' },
    { label: 'Loss %', data: [], backgroundColor: color('#e05c6a', 0.8), stack: 's' },
  ], { scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, max: 100, stacked: true }, x: { ...CHART_DEFAULTS.scales.x, stacked: true } } });

  charts.entropy = mkChart('c-entropy', 'line', [
    { label: 'Opening Entropy (bits)', data: [], borderColor: '#7c6af7', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y' },
    { label: 'Top-1 Freq', data: [], borderColor: '#e0a84a', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y2', borderDash: [4,3] },
  ], {
    scales: {
      x: CHART_DEFAULTS.scales.x,
      y:  { ...CHART_DEFAULTS.scales.y, position: 'left' },
      y2: { ...CHART_DEFAULTS.scales.y, position: 'right', grid: { drawOnChartArea: false } },
    },
  });

  charts.enEntropy = mkChart('c-en-entropy', 'line', [
    { label: 'Eval Entropy (bits)', data: [], borderColor: '#5ee8c0', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y' },
    { label: 'Eval Top-1 Freq', data: [], borderColor: '#e05c6a', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y2', borderDash: [4,3] },
  ], {
    scales: {
      x: CHART_DEFAULTS.scales.x,
      y:  { ...CHART_DEFAULTS.scales.y, position: 'left' },
      y2: { ...CHART_DEFAULTS.scales.y, position: 'right', grid: { drawOnChartArea: false } },
    },
  });

  charts.gamelen = mkChart('c-gamelen', 'line', [
    { label: 'Avg Game Len', data: [], borderColor: '#5ee8c0', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y' },
    { label: 'Avg Walls Placed', data: [], borderColor: '#e05c6a', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y2', borderDash: [4,3] },
  ], {
    scales: {
      x: CHART_DEFAULTS.scales.x,
      y:  { ...CHART_DEFAULTS.scales.y, position: 'left' },
      y2: { ...CHART_DEFAULTS.scales.y, position: 'right', grid: { drawOnChartArea: false } },
    },
  });

  charts.resign = mkChart('c-resign', 'line', [
    { label: 'Resign %',          data: [], borderColor: '#e05c6a', backgroundColor: color('#e05c6a', 0.1), fill: true, tension: 0.3, yAxisID: 'y' },
    { label: 'False-Resign Count', data: [], borderColor: '#e0a84a', backgroundColor: 'transparent', tension: 0.3, yAxisID: 'y2', borderDash: [4,3] },
  ], {
    scales: {
      x: CHART_DEFAULTS.scales.x,
      y:  { ...CHART_DEFAULTS.scales.y, position: 'left',  title: { display: true, text: 'Resign %', color: '#aaa' } },
      y2: { ...CHART_DEFAULTS.scales.y, position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'False-Resign', color: '#aaa' } },
    },
  });

  charts.eval = mkChart('c-eval', 'line', [
    { label: 'Eval Score', data: [], borderColor: '#5ee8c0', backgroundColor: color('#5ee8c0', 0.1), fill: true, tension: 0.3 },
    { label: 'Promote Threshold', data: [], borderColor: '#e05c6a', borderDash: [6,3], pointRadius: 0 },
  ]);

  charts.loss = mkChart('c-loss', 'line', [
    { label: 'Total Loss',  data: [], borderColor: '#7c6af7', backgroundColor: 'transparent', tension: 0.3 },
    { label: 'Policy Loss', data: [], borderColor: '#5ee8c0', backgroundColor: 'transparent', tension: 0.3, borderDash: [4,2] },
    { label: 'Value Loss',  data: [], borderColor: '#e0a84a', backgroundColor: 'transparent', tension: 0.3, borderDash: [4,2] },
  ]);

  charts.bench = mkChart('c-bench', 'line', [
    { label: 'vs Random', data: [], borderColor: '#4caf7d', backgroundColor: 'transparent', tension: 0.3 },
    { label: 'vs Greedy', data: [], borderColor: '#e0a84a', backgroundColor: 'transparent', tension: 0.3 },
    { label: 'vs BFS',    data: [], borderColor: '#e05c6a', backgroundColor: 'transparent', tension: 0.3 },
  ]);

  charts.timing = mkChart('c-timing', 'bar', [
    { label: 'Self-Play', data: [], backgroundColor: color('#7c6af7', 0.8), stack: 's' },
    { label: 'Train',     data: [], backgroundColor: color('#5ee8c0', 0.8), stack: 's' },
    { label: 'Eval Net',  data: [], backgroundColor: color('#e0a84a', 0.8), stack: 's' },
    { label: 'Eval Best', data: [], backgroundColor: color('#e05c6a', 0.8), stack: 's' },
  ], { scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y }, x: { ...CHART_DEFAULTS.scales.x, stacked: true } } });
}

// ── Data loading ───────────────────────────────────────────────────────

function updateSummary(rows) {
  if (!rows.length) return;
  const last = rows[rows.length - 1];
  document.getElementById('s-cycles').textContent  = rows.length;

  const totalMin = rows.reduce((acc, r) => {
    return acc +
      (parseFloat(r.t_selfplay_min) || 0) +
      (parseFloat(r.t_train_min)    || 0) +
      (parseFloat(r.t_evalnet_min)  || 0) +
      (parseFloat(r.t_evalbest_min) || 0);
  }, 0);
  if (totalMin > 0) {
    const h = Math.floor(totalMin / 60);
    const m = Math.round(totalMin % 60);
    document.getElementById('s-total-time').textContent = h > 0 ? `${h}h ${m}m` : `${m}m`;
  } else {
    document.getElementById('s-total-time').textContent = '—';
  }

  const lastCycleMin =
    (parseFloat(last.t_selfplay_min) || 0) +
    (parseFloat(last.t_train_min)    || 0) +
    (parseFloat(last.t_evalnet_min)  || 0) +
    (parseFloat(last.t_evalbest_min) || 0);
  if (lastCycleMin > 0) {
    const lh = Math.floor(lastCycleMin / 60);
    const lm = Math.round(lastCycleMin % 60);
    document.getElementById('s-cycle-time').textContent = lh > 0 ? `${lh}h ${lm}m` : `${lm}m`;
  } else {
    document.getElementById('s-cycle-time').textContent = '—';
  }

  document.getElementById('s-latest').textContent  = last.cycle ?? '—';
  document.getElementById('s-eval').textContent    = last.eval_score != null ? last.eval_score.toFixed(3) : '—';
  document.getElementById('s-loss').textContent    = last.loss != null ? last.loss.toFixed(4) : '—';
  document.getElementById('s-bfs').textContent     = last.vs_bfs    != null ? (last.vs_bfs * 100).toFixed(0)    + '%' : '—';
  document.getElementById('s-greedy').textContent  = last.vs_greedy != null ? (last.vs_greedy * 100).toFixed(0) + '%' : '—';
  document.getElementById('s-gamelen').textContent = last.sp_avg_game_len != null ? last.sp_avg_game_len.toFixed(1) : '—';
  document.getElementById('s-entropy').textContent = last.sp_entropy != null ? last.sp_entropy.toFixed(2) : '—';
  document.getElementById('s-resign').textContent = last.sp_resign_pct != null ? last.sp_resign_pct.toFixed(1) + '%' : '—';
  document.getElementById('s-false-resign').textContent = last.sp_false_resign != null ? last.sp_false_resign : '—';
}

function updateCharts(rows) {
  const cycles = rows.map(r => r.cycle);

  function setData(chart, labels, ...datasets) {
    chart.data.labels = labels;
    datasets.forEach((d, i) => { if (chart.data.datasets[i]) chart.data.datasets[i].data = d; });
    chart.update();
  }

  setData(charts.outcomes, cycles,
    rows.map(r => r.sp_W_pct),
    rows.map(r => r.sp_D_pct),
    rows.map(r => r.sp_L_pct),
  );

  setData(charts.entropy, cycles,
    rows.map(r => r.sp_entropy),
    rows.map(r => r.sp_top1),
  );

  setData(charts.enEntropy, cycles,
    rows.map(r => r.en_entropy  != null && r.en_entropy  !== '' ? r.en_entropy  : null),
    rows.map(r => r.en_top1     != null && r.en_top1     !== '' ? r.en_top1     : null),
  );

  setData(charts.gamelen, cycles,
    rows.map(r => r.sp_avg_game_len),
    rows.map(r => r.sp_avg_walls),
  );

  setData(charts.resign, cycles,
    rows.map(r => r.sp_resign_pct  != null && r.sp_resign_pct  !== '' ? r.sp_resign_pct  : null),
    rows.map(r => r.sp_false_resign != null && r.sp_false_resign !== '' ? r.sp_false_resign : null),
  );

  setData(charts.eval, cycles,
    rows.map(r => r.eval_score),
    rows.map(() => 0.55),   // EN_PROMOTE_THRESHOLD default
  );

  setData(charts.loss, cycles,
    rows.map(r => r.loss),
    rows.map(r => r.loss_policy),
    rows.map(r => r.loss_value),
  );

  setData(charts.bench, cycles,
    rows.map(r => r.vs_random),
    rows.map(r => r.vs_greedy),
    rows.map(r => r.vs_bfs),
  );

  setData(charts.timing, cycles,
    rows.map(r => r.t_selfplay_min),
    rows.map(r => r.t_train_min),
    rows.map(r => r.t_evalnet_min),
    rows.map(r => r.t_evalbest_min),
  );
}

async function loadStats() {
  const resp = await fetch('/api/stats');
  const rows = await resp.json();
  updateSummary(rows);
  updateCharts(rows);
  updateStatsTable(rows);
  document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
}

// ── Stats history table ───────────────────────────────────────────────

function updateStatsTable(rows) {
  const el = document.getElementById('stats-table-inner');
  if (!rows.length) { el.innerHTML = '<em style="color:var(--text-dim)">No data yet.</em>'; return; }

  // colour helpers
  function lerpRGB(a, b, t) {
    return [Math.round(a[0]+(b[0]-a[0])*t), Math.round(a[1]+(b[1]-a[1])*t), Math.round(a[2]+(b[2]-a[2])*t)];
  }
  function rdylgn(t) { // red→yellow→green, t in [0,1]
    t = Math.max(0, Math.min(1, t));
    const red=[215,48,39], yel=[255,255,191], grn=[26,152,80];
    const [r,g,b] = t < 0.5 ? lerpRGB(red, yel, t*2) : lerpRGB(yel, grn, (t-0.5)*2);
    return { bg: `rgb(${r},${g},${b})`, fg: (0.299*r+0.587*g+0.114*b) < 128 ? '#f1f1f1' : '#000' };
  }
  function blues(t) { // white→dark-blue
    t = Math.max(0, Math.min(1, t));
    const [r,g,b] = lerpRGB([247,251,255], [8,48,107], t);
    return { bg: `rgb(${r},${g},${b})`, fg: (0.299*r+0.587*g+0.114*b) < 128 ? '#f1f1f1' : '#000' };
  }

  const COLS = [
    { key: 'cycle',           hdr: 'Cycle',    fmt: v => v },
    { key: 't_cumulative_h',  hdr: 'Total h',  fmt: v => v != null ? v.toFixed(1)+'h' : '—' },
    { key: 'sp_resign_pct',   hdr: 'Resign%',  fmt: v => v != null ? v.toFixed(1)+'%' : '—' },
    { key: 'sp_false_resign', hdr: 'FalseRes', fmt: v => v != null ? v : '—' },
    { key: 'sp_entropy',      hdr: 'Entropy',  fmt: v => v != null ? v.toFixed(2) : '—', color: v => v != null ? rdylgn((v-5)/3) : null },
    { key: 'sp_top1',         hdr: 'Top-1',    fmt: v => v != null ? v : '—' },
    { key: 'sp_avg_game_len', hdr: 'Avg Plies',fmt: v => v != null ? v.toFixed(1) : '—', color: v => v != null ? blues((v-10)/30) : null },
    { key: 'sp_avg_walls',    hdr: 'Walls',    fmt: v => v != null ? v.toFixed(1) : '—', color: v => v != null ? blues(v/10) : null },
    { key: 'loss',            hdr: 'Loss',     fmt: v => v != null ? v.toFixed(4) : '—' },
    { key: 'loss_value',      hdr: 'Val Loss', fmt: v => v != null ? v.toFixed(4) : '—' },
    { key: 'eval_score',      hdr: 'Eval',     fmt: v => v != null ? v.toFixed(4) : '—', color: v => v != null ? rdylgn(v) : null },
    { key: 'promoted',        hdr: '★',        fmt: v => v ? '★' : '' },
    { key: 'vs_greedy',       hdr: 'vs Greedy',fmt: v => v != null ? v.toFixed(2) : '—', color: v => v != null ? rdylgn(v) : null },
    { key: 'vs_bfs',          hdr: 'vs BFS',   fmt: v => v != null ? v.toFixed(2) : '—', color: v => v != null ? rdylgn(v) : null },
    { key: 't_cycle_min',     hdr: 'Min',      fmt: v => v != null ? v.toFixed(1) : '—' },
  ];

  // compute cumulative time so we can show it
  let cumH = 0;
  const enriched = rows.map(r => {
    cumH += (parseFloat(r.t_cycle_min) || 0) / 60;
    return { ...r, t_cumulative_h: cumH };
  });

  // only show columns that have at least one non-null value
  const visible = COLS.filter(c => enriched.some(r => r[c.key] != null && r[c.key] !== ''));

  let html = '<div class="stats-table-scroll"><table class="stats-history-table"><thead><tr>';
  for (const c of visible) html += `<th>${c.hdr}</th>`;
  html += '</tr></thead><tbody>';

  for (const r of enriched) {
    html += '<tr>';
    for (const c of visible) {
      const v = r[c.key] != null && r[c.key] !== '' ? (typeof r[c.key] === 'string' ? parseFloat(r[c.key]) || r[c.key] : r[c.key]) : null;
      const col = c.color ? c.color(typeof v === 'string' ? null : v) : null;
      const style = col ? ` style="background:${col.bg};color:${col.fg}"` : '';
      html += `<td${style}>${c.fmt(typeof v === 'string' ? v : v)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  el.innerHTML = html;

  // scroll to bottom so latest cycle is visible by default
  const scrollEl = el.querySelector('.stats-table-scroll');
  if (scrollEl) scrollEl.scrollTop = scrollEl.scrollHeight;
}

// ── Training control ──────────────────────────────────────────────────

async function loadTrainingStatus() {
  try {
    const resp = await fetch('/api/training/status');
    const d = await resp.json();
    const badge = document.getElementById('tc-badge');
    const cycleLabel = document.getElementById('tc-cycle');
    const btnStart = document.getElementById('btn-train-start');
    const btnStop  = document.getElementById('btn-train-stop');
    const btnKill  = document.getElementById('btn-train-kill');

    if (d.running && d.stop_pending) {
      badge.textContent = 'Stopping…';
      badge.className = 'badge badge-stopping';
    } else if (d.running) {
      badge.textContent = 'Running';
      badge.className = 'badge badge-running';
    } else {
      badge.textContent = 'Idle';
      badge.className = 'badge badge-idle';
    }
    cycleLabel.textContent = d.last_cycle != null ? `· last cycle ${d.last_cycle}` : '';
    btnStart.disabled = d.running;
    btnStop.disabled  = !d.running || d.stop_pending;
    btnKill.disabled  = !d.running;
    const logEl = document.getElementById('tc-log');
    if (logEl) logEl.textContent = d.last_log_line || ''
  } catch (e) { /* server may be restarting */ }
}

async function startTraining() {
  await fetch('/api/training/start', { method: 'POST' });
  loadTrainingStatus();
}

async function stopTraining() {
  await fetch('/api/training/stop', { method: 'POST' });
  loadTrainingStatus();
}

async function killTraining() {
  if (!confirm('Kill the training process immediately?')) return;
  await fetch('/api/training/kill', { method: 'POST' });
  loadTrainingStatus();
}

// ── Init ───────────────────────────────────────────────────────────────

initCharts();
loadStats();
loadTrainingStatus();
setInterval(loadStats, 30_000);
setInterval(loadTrainingStatus, 5_000);
