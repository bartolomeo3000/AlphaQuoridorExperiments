/**
 * game_state.js — Lightweight JS port of State.next() for client-side replay.
 *
 * Only tracks positions, wall placement, and perspective swap.
 * No legality checking, no BFS — just enough to reconstruct the board
 * for every step of an action-sequence replay.
 *
 * Convention (matches Python State):
 *   - player/enemy positions are stored in the CURRENT PLAYER's frame.
 *   - player[0] = position (row*N+col), row 0 = that player's GOAL row.
 *   - walls[i] = 0|1|2  (0=empty, 1=h-wall, 2=v-wall)  (N-1)² entries.
 *   - After next(), rotate_walls() then swap player/enemy.
 *
 * Public API:
 *   const gs = new GameState(N, numWalls);  // fresh starting state
 *   const gs2 = gs.next(action);            // immutable — returns new state
 *   gs.toDisplay()                           // → display dict for QuoridorBoard
 */

class GameState {
  /**
   * @param {number} N          Board size (7)
   * @param {number[]} player   [pos, wallsLeft]
   * @param {number[]} enemy    [pos, wallsLeft]
   * @param {number[]} walls    (N-1)² ints
   * @param {number}   depth
   */
  constructor(N, player, enemy, walls, depth) {
    this.N     = N;
    this.player = player.slice();
    this.enemy  = enemy.slice();
    this.walls  = walls.slice();
    this.depth  = depth || 0;
  }

  /** Create the standard starting state. */
  static initial(N = 7, numWalls = 5) {
    const initPos = N * (N - 1) + Math.floor(N / 2);  // bottom-centre
    return new GameState(
      N,
      [initPos, numWalls],
      [initPos, numWalls],
      new Array((N - 1) * (N - 1)).fill(0),
      0,
    );
  }

  /** Apply an action and return a new GameState. */
  next(action) {
    const N = this.N;
    const np = this.player.slice();
    const ne = this.enemy.slice();
    const nw = this.walls.slice();

    if (action < N * N) {
      // Move pawn
      np[0] = action;
    } else if (action < N * N + (N - 1) * (N - 1)) {
      // Place horizontal wall
      const pos = action - N * N;
      nw[pos] = 1;
      np[1] -= 1;
    } else {
      // Place vertical wall
      const pos = action - N * N - (N - 1) * (N - 1);
      nw[pos] = 2;
      np[1] -= 1;
    }

    // rotate_walls: reverse the array (180° board rotation)
    const rw = nw.slice().reverse();

    // Swap player/enemy
    return new GameState(N, ne, np, rw, this.depth + 1);
  }

  /** Is this P1's turn (depth even)? */
  isFirstPlayer() {
    return this.depth % 2 === 0;
  }

  /**
   * Convert to the display dict expected by QuoridorBoard.draw().
   *
   * Absolute frame: P1 starts at row N-1 (bottom), goal = row 0 (top).
   *                 P2 starts at row 0  (top),    goal = row N-1 (bottom).
   * Internal frame: current player always starts at row N-1.
   */
  toDisplay() {
    const N = this.N;
    const fp = this.isFirstPlayer();

    let p1_pos, p2_pos, p1_walls, p2_walls, walls_abs;

    if (fp) {
      // depth even → player slot = P1 (direct), enemy slot = P2 (flipped)
      p1_pos   = this.player[0];
      p2_pos   = N * N - 1 - this.enemy[0];
      p1_walls = this.player[1];
      p2_walls = this.enemy[1];
      walls_abs = this.walls.slice();
    } else {
      // depth odd → player slot = P2 (direct, but in their rotated frame),
      //             enemy slot = P1 (flipped)
      p1_pos   = this.enemy[0];          // P1 stored in enemy slot, already absolute for even step
      p2_pos   = N * N - 1 - this.player[0];
      p1_walls = this.enemy[1];
      p2_walls = this.player[1];
      walls_abs = this.walls.slice().reverse();  // un-rotate to P1-absolute
    }

    return {
      N,
      p1_pos,
      p2_pos,
      p1_walls_left: p1_walls,
      p2_walls_left: p2_walls,
      walls: walls_abs,
      depth: this.depth,
      is_p1_turn: fp,
      legal: [],   // not computed client-side
      done: false,
      winner: null,
    };
  }

  /**
   * Replay a full action sequence from the start,
   * returning an array of display states (length = actions.length + 1).
   */
  static replayActions(actions, N = 7, numWalls = 5) {
    const states = [];
    let s = GameState.initial(N, numWalls);
    states.push(s.toDisplay());
    for (const a of actions) {
      s = s.next(a);
      states.push(s.toDisplay());
    }
    return states;
  }
}
