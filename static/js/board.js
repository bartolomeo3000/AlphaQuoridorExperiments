/**
 * board.js — Shared Canvas-based Quoridor board renderer.
 *
 * Usage:
 *   const b = new QuoridorBoard(canvasEl, { N: 7, cellSize: 64, wallSize: 8, gap: 4 });
 *   b.draw(displayState);          // full redraw
 *   b.onClick(handler);            // handler(type, payload)
 *       type = 'cell'   → payload = { pos, row, col }
 *       type = 'hwall'  → payload = { wpos }   (wall-grid index)
 *       type = 'vwall'  → payload = { wpos }
 *
 * displayState (from server or computed locally):
 *   { N, p1_pos, p2_pos, p1_walls_left, p2_walls_left,
 *     walls: int[36],   // 0=empty, 1=h-wall, 2=v-wall  (P1-absolute frame)
 *     is_p1_turn, legal: int[], done, winner,
 *     highlight?: Set<int>,   // optional cell highlights
 *     lastMove?: int,          // optional last-moved cell/action
 *   }
 */

class QuoridorBoard {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.N      = opts.N       || 7;
    this.CS     = opts.cellSize || 64;   // cell size px
    this.WS     = opts.wallSize || 8;    // wall thickness px
    this.GAP    = opts.gap      || 4;    // gap between cells / walls

    this._clickHandlers = [];
    this._state = null;
    this._overlayMode = null;  // 'hwall' | 'vwall' | null — hover intent

    this._resize();
    canvas.addEventListener('mousemove', e => this._onHover(e));
    canvas.addEventListener('mouseleave', () => { this._hoverInfo = null; if (this._state) this.draw(this._state); });
    canvas.addEventListener('click', e => this._onClickRaw(e));
  }

  _resize() {
    const N = this.N, CS = this.CS, WS = this.WS, GAP = this.GAP;
    const LABEL = 20;  // px reserved on each side for coordinate labels
    const inner = N * CS + (N - 1) * (WS + GAP);
    const total = inner + LABEL * 2;
    this.canvas.width  = total;
    this.canvas.height = total;
    this.canvas.style.width  = total + 'px';
    this.canvas.style.height = total + 'px';
    this._total = total;
    this._LABEL = LABEL;
  }

  // ── coordinate helpers ───────────────────────────────────────────────

  /** Top-left pixel of cell (row, col) */
  _cellXY(row, col) {
    const step = this.CS + this.WS + this.GAP;
    const x = this._LABEL + col * step;
    const y = this._LABEL + row * step;
    return { x, y };
  }

  /** Board position index → {row, col} */
  _posToRC(pos) {
    return { row: Math.floor(pos / this.N), col: pos % this.N };
  }

  /** Canvas pixel → hit-test { type, rc, wrow, wcol }
   *  overlayMode: 'hwall' | 'vwall' | 'cell' | null
   *  When a wall mode is active, wall slots switch at cell midpoints so the
   *  user can click the centre of the gap rather than a precise thin strip.
   */
  _hitTest(px, py, overlayMode) {
    const N = this.N, CS = this.CS, WS = this.WS, GAP = this.GAP;
    const LABEL = this._LABEL;
    const step = CS + WS + GAP;
    const rx = px - LABEL;
    const ry = py - LABEL;

    if (overlayMode === 'hwall') {
      // Row: must be in the gap strip between rows (cursor below the cell area)
      const wr = Math.floor(ry / step);
      const ry_frac = ry - wr * step;
      if (wr >= 0 && wr < N - 1 && ry_frac > CS) {
        // Column: shift right by half a cell so the slot switches at midpoints
        const wc = Math.floor((rx - CS / 2) / step);
        if (wc >= 0 && wc < N - 1) return { type: 'hwall', wrow: wr, wcol: wc };
      }
      // Not in gap strip — fall through to cell detection
    }

    if (overlayMode === 'vwall') {
      // Col: must be in the gap strip between cols (cursor right of the cell area)
      const wc = Math.floor(rx / step);
      const rx_frac = rx - wc * step;
      if (wc >= 0 && wc < N - 1 && rx_frac > CS) {
        // Row: shift up by half a cell so the slot switches at midpoints
        const wr = Math.floor((ry - CS / 2) / step);
        if (wr >= 0 && wr < N - 1) return { type: 'vwall', wrow: wr, wcol: wc };
      }
      // Not in gap strip — fall through to cell detection
    }

    // Default: only hit cells (not gap/wall strips)
    const ci = Math.floor(rx / step), ri = Math.floor(ry / step);
    if (ci < 0 || ri < 0 || ci >= N || ri >= N) return null;
    const cf = (rx / step) - ci, rf = (ry / step) - ri;
    if (cf * step <= CS && rf * step <= CS) {
      return { type: 'cell', row: ri, col: ci };
    }
    return null;
  }

  onOverlayMode(mode) {
    this._overlayMode = mode;  // 'hwall' | 'vwall' | 'cell' | null
  }

  /** Highlight an action from an external source (e.g. policy bar hover).
   *  info: { type:'cell', row, col } | { type:'hwall'|'vwall', wrow, wcol } | null
   */
  setExternalHover(info) {
    this._externalHover = info;
    if (this._state) this.draw(this._state);
  }

  clearExternalHover() {
    this._externalHover = null;
    if (this._state) this.draw(this._state);
  }

  onClick(handler) {
    this._clickHandlers.push(handler);
  }

  _onClickRaw(e) {
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const hit = this._hitTest(px, py, this._overlayMode);
    if (!hit) return;

    for (const h of this._clickHandlers) {
      if (hit.type === 'cell') {
        h('cell', { pos: hit.row * this.N + hit.col, row: hit.row, col: hit.col });
      } else if (hit.type === 'hwall') {
        h('hwall', { wpos: hit.wrow * (this.N - 1) + hit.wcol, wrow: hit.wrow, wcol: hit.wcol });
      } else if (hit.type === 'vwall') {
        h('vwall', { wpos: hit.wrow * (this.N - 1) + hit.wcol, wrow: hit.wrow, wcol: hit.wcol });
      }
    }
  }

  _onHover(e) {
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const hit = this._hitTest(px, py, this._overlayMode);
    this._hoverInfo = hit;
    if (this._state) this.draw(this._state);
  }

  // ── drawing ──────────────────────────────────────────────────────────

  draw(state) {
    this._state = state;
    const ctx = this.ctx;
    const N = this.N, CS = this.CS, WS = this.WS;

    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Background
    ctx.fillStyle = '#141620';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    const legalSet = new Set(state.legal || state.legal_actions || []);
    const highlight = state.highlight || new Set();

    // Draw cells
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const pos = r * N + c;
        const { x, y } = this._cellXY(r, c);

        // Cell fill
        let fill = '#1e2133';
        if (highlight.has && highlight.has(pos)) fill = 'rgba(94,232,192,0.18)';
        if (legalSet.has(pos)) fill = 'rgba(124,106,247,0.22)';

        ctx.fillStyle = fill;
        this._roundRect(ctx, x, y, CS, CS, 4);
        ctx.fill();

        // Hover outline on legal cells
        if (this._isHovering('cell', r, c) && legalSet.has(pos)) {
          ctx.strokeStyle = 'rgba(94,232,192,0.85)';
          ctx.lineWidth = 2;
          this._roundRect(ctx, x, y, CS, CS, 4);
          ctx.stroke();
        }

        // Row labels — left edge
        if (c === 0) {
          ctx.fillStyle = '#6b7299';
          ctx.font = 'bold 11px monospace';
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(r), x - 6, y + CS / 2);
        }
        // Row labels — right edge
        if (c === N - 1) {
          ctx.fillStyle = '#6b7299';
          ctx.font = 'bold 11px monospace';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(r), x + CS + 6, y + CS / 2);
        }
        // Col labels — top edge
        if (r === 0) {
          ctx.fillStyle = '#6b7299';
          ctx.font = 'bold 11px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(String(c), x + CS / 2, y - 5);
        }
        // Col labels — bottom edge
        if (r === N - 1) {
          ctx.fillStyle = '#6b7299';
          ctx.font = 'bold 11px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText(String(c), x + CS / 2, y + CS + 5);
        }
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
      }
    }

    // Draw placed walls
    const walls = state.walls || [];
    for (let wr = 0; wr < N - 1; wr++) {
      for (let wc = 0; wc < N - 1; wc++) {
        const wpos = wr * (N - 1) + wc;
        const w = walls[wpos];
        if (w === 1) this._drawHWall(ctx, wr, wc, '#e0a84a', 1.0);
        if (w === 2) this._drawVWall(ctx, wr, wc, '#7c6af7', 1.0);
      }
    }

    // Draw wall hover preview (mouse hover)
    const h = this._hoverInfo;
    if (h && this._overlayMode === 'hwall' && h.type === 'hwall') {
      this._drawHWall(ctx, h.wrow, h.wcol, '#e0a84a', 0.45);
    }
    if (h && this._overlayMode === 'vwall' && h.type === 'vwall') {
      this._drawVWall(ctx, h.wrow, h.wcol, '#7c6af7', 0.45);
    }

    // Draw external hover (policy bar hover)
    const eh = this._externalHover;
    if (eh) {
      if (eh.type === 'hwall') {
        this._drawHWall(ctx, eh.wrow, eh.wcol, '#e0a84a', 0.65);
      } else if (eh.type === 'vwall') {
        this._drawVWall(ctx, eh.wrow, eh.wcol, '#7c6af7', 0.65);
      } else if (eh.type === 'cell') {
        const { x, y } = this._cellXY(eh.row, eh.col);
        ctx.strokeStyle = 'rgba(94,232,192,0.9)';
        ctx.lineWidth = 3;
        this._roundRect(ctx, x, y, CS, CS, 4);
        ctx.stroke();
        // also fill lightly
        ctx.fillStyle = 'rgba(94,232,192,0.18)';
        this._roundRect(ctx, x, y, CS, CS, 4);
        ctx.fill();
      }
    }

    // Draw last-move highlight
    if (state.lastMove != null && state.lastMove < N * N) {
      const { row, col } = this._posToRC(state.lastMove);
      const { x, y } = this._cellXY(row, col);
      ctx.strokeStyle = 'rgba(255,220,80,0.6)';
      ctx.lineWidth = 2;
      this._roundRect(ctx, x, y, CS, CS, 4);
      ctx.stroke();
    }

    // Draw P2 (top pawn, blue)
    if (state.p2_pos != null) {
      const { row, col } = this._posToRC(state.p2_pos);
      const { x, y } = this._cellXY(row, col);
      this._drawPawn(ctx, x + CS / 2, y + CS / 2, CS * 0.36, '#5ea8f7', state.p2_walls_left);
    }

    // Draw P1 (bottom pawn, green)
    if (state.p1_pos != null) {
      const { row, col } = this._posToRC(state.p1_pos);
      const { x, y } = this._cellXY(row, col);
      this._drawPawn(ctx, x + CS / 2, y + CS / 2, CS * 0.36, '#5ee8c0', state.p1_walls_left);
    }

    // Win/draw overlay
    if (state.done && state.winner) {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.fillStyle = state.winner === 'draw' ? '#e0a84a' :
                      state.winner === 'p1'   ? '#5ee8c0' : '#5ea8f7';
      ctx.font = `bold ${CS * 0.7}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const msg = state.winner === 'draw' ? 'DRAW' :
                  state.winner === 'p1'   ? 'P1 WINS' : 'P2 WINS';
      ctx.fillText(msg, this.canvas.width / 2, this.canvas.height / 2);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
    }
  }

  _drawHWall(ctx, wr, wc, color, alpha) {
    const N = this.N, CS = this.CS, WS = this.WS, GAP = this.GAP, LABEL = this._LABEL;
    const step = CS + WS + GAP;
    // A horizontal wall spans cells (wr,wc) and (wr,wc+1) vertically between rows wr and wr+1
    const x = LABEL + wc * step;
    const y = LABEL + wr * step + CS + GAP;
    const w = CS * 2 + WS + GAP;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    this._roundRect(ctx, x, y, w, WS, 3);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1.0;
  }

  _drawVWall(ctx, wr, wc, color, alpha) {
    const N = this.N, CS = this.CS, WS = this.WS, GAP = this.GAP, LABEL = this._LABEL;
    const step = CS + WS + GAP;
    // A vertical wall spans cells (wr,wc) and (wr+1,wc) horizontally between cols wc and wc+1
    const x = LABEL + wc * step + CS + GAP;
    const y = LABEL + wr * step;
    const h = CS * 2 + WS + GAP;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    this._roundRect(ctx, x, y, WS, h, 3);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1.0;
  }

  _drawPawn(ctx, cx, cy, r, color, wallsLeft) {
    ctx.shadowColor = color;
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Border ring
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Wall count label inside pawn
    ctx.fillStyle = '#0a0c12';
    ctx.font = `bold ${Math.round(r * 0.9)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(wallsLeft != null ? wallsLeft : ''), cx, cy);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  _isHovering(type, r, c) {
    const h = this._hoverInfo;
    if (!h || h.type !== type) return false;
    if (type === 'cell') return h.row === r && h.col === c;
    return h.wrow === r && h.wcol === c;
  }

  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
  }

  /**
   * Draw a 7×7 heatmap onto a small canvas.
   * values: 2D array [row][col], numeric.
   * opts: { colorFn, showNums, cellSize }
   */
  static drawHeatmap(canvas, values, opts = {}) {
    const N = values.length;
    const CS = opts.cellSize || 28;
    canvas.width  = N * CS;
    canvas.height = N * CS;
    canvas.style.width  = (N * CS) + 'px';
    canvas.style.height = (N * CS) + 'px';

    const ctx  = canvas.getContext('2d');
    const flat = values.flat();
    const finite = flat.filter(v => v < 9999);
    const mn = finite.length ? Math.min(...finite) : 0;
    const mx = finite.length ? Math.max(...finite) : 1;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const v = values[r][c];
        const fill = opts.colorFn
          ? opts.colorFn(v, mn, mx)
          : QuoridorBoard._heatColor(v, mn, mx);
        ctx.fillStyle = fill;
        ctx.fillRect(c * CS, r * CS, CS, CS);

        // Numeric label
        if (opts.showNums !== false) {
          const t = (v >= 9999) ? '∞' : String(v);
          const bright = QuoridorBoard._luminance(fill) < 0.45;
          ctx.fillStyle = bright ? '#fff' : '#111';
          ctx.font = `bold ${Math.round(CS * 0.38)}px monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(t, c * CS + CS / 2, r * CS + CS / 2);
        }
      }
    }
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  /** Map value v in [mn, mx] → CSS color string (green → yellow → red). */
  static _heatColor(v, mn, mx) {
    if (v >= 9999) return '#1a1a2e';
    const t = mx > mn ? (v - mn) / (mx - mn) : 0;
    // green (0,180,100) → yellow (220,200,0) → red (220,40,40)
    let r, g, b;
    if (t < 0.5) {
      r = Math.round(t * 2 * 220);
      g = Math.round(180 + t * 2 * 20);
      b = Math.round(100 - t * 2 * 100);
    } else {
      const s = (t - 0.5) * 2;
      r = 220;
      g = Math.round(200 - s * 160);
      b = Math.round(s * 0);
    }
    return `rgb(${r},${g},${b})`;
  }

  static _luminance(cssColor) {
    // rough luminance from rgb(...) string
    const m = cssColor.match(/\d+/g);
    if (!m) return 0.5;
    return (parseInt(m[0]) * 0.299 + parseInt(m[1]) * 0.587 + parseInt(m[2]) * 0.114) / 255;
  }
}
