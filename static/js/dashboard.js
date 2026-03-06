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
  document.getElementById('s-latest').textContent  = last.cycle ?? '—';
  document.getElementById('s-eval').textContent    = last.eval_score != null ? last.eval_score.toFixed(3) : '—';
  document.getElementById('s-loss').textContent    = last.loss != null ? last.loss.toFixed(4) : '—';
  document.getElementById('s-bfs').textContent     = last.vs_bfs    != null ? (last.vs_bfs * 100).toFixed(0)    + '%' : '—';
  document.getElementById('s-greedy').textContent  = last.vs_greedy != null ? (last.vs_greedy * 100).toFixed(0) + '%' : '—';
  document.getElementById('s-gamelen').textContent = last.sp_avg_game_len != null ? last.sp_avg_game_len.toFixed(1) : '—';
  document.getElementById('s-entropy').textContent = last.sp_entropy != null ? last.sp_entropy.toFixed(2) : '—';
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

  setData(charts.gamelen, cycles,
    rows.map(r => r.sp_avg_game_len),
    rows.map(r => r.sp_avg_walls),
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
  document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
}

// ── Init ───────────────────────────────────────────────────────────────

initCharts();
loadStats();
setInterval(loadStats, 30_000);
