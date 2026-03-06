/* logs.js — SSE-powered live log viewer */

const terminal   = document.getElementById('log-terminal');
const statusEl   = document.getElementById('sse-status');
const autoScroll = document.getElementById('autoscroll');

const MAX_LINES = 500;
let lineCount   = 0;
let sse         = null;

function clearLog() {
  terminal.innerHTML = '';
  lineCount = 0;
}

function appendLine(text, backfill) {
  const span = document.createElement('span');
  span.className = classifyLine(text);
  span.textContent = text + '\n';
  if (backfill) span.style.opacity = '0.65';
  terminal.appendChild(span);
  lineCount++;

  // Prune old lines beyond MAX_LINES
  while (terminal.children.length > MAX_LINES) {
    terminal.removeChild(terminal.firstChild);
  }

  if (autoScroll.checked) {
    terminal.scrollTop = terminal.scrollHeight;
  }
}

function classifyLine(text) {
  if (/={3,}\s*Cycle\s+\d+/.test(text))  return 'log-cycle';
  if (/\[timing\]/.test(text))            return 'log-timing';
  if (/error|exception|traceback/i.test(text)) return 'log-error';
  return '';
}

function connect() {
  if (sse) { sse.close(); sse = null; }

  statusEl.textContent  = '● Connecting…';
  statusEl.style.color  = 'var(--text-dim)';

  sse = new EventSource('/api/logs/stream');

  sse.onopen = () => {
    statusEl.textContent = '● Connected';
    statusEl.style.color = 'var(--win)';
  };

  sse.onmessage = (e) => {
    try {
      const { text, backfill } = JSON.parse(e.data);
      appendLine(text, !!backfill);
    } catch (err) {
      appendLine(e.data, false);
    }
  };

  sse.onerror = () => {
    statusEl.textContent = '● Disconnected — retrying…';
    statusEl.style.color = 'var(--loss)';
    sse.close();
    sse = null;
    setTimeout(connect, 3000);
  };
}

// ── Init ───────────────────────────────────────────────────────────────
connect();
