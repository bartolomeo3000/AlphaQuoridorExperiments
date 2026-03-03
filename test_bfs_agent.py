"""
Quick sanity tests for bfs_forward_action.
Run with: python test_bfs_agent.py
"""
import sys, traceback
from collections import deque
from game import State
from evaluate_best_player import bfs_forward_action, greedy_forward_action

PASS = "PASS"
FAIL = "FAIL"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

# ── helpers ──────────────────────────────────────────────────────────────────

def place_wall_on(state, wall_index):
    """Return a new state with the given wall placed, or None if illegal."""
    if wall_index in state.legal_actions():
        return state.next(wall_index)
    return None

# ── Test 1: open board — returns a valid pawn move ────────────────────────────
try:
    state = State()
    N = state.N
    action = bfs_forward_action(state)
    legal = state.legal_actions()
    check("open board: action is legal", action in legal, f"action={action}, legal={legal}")
    check("open board: action is pawn move", action < N * N, f"action={action}")
except Exception as e:
    check("open board", False, traceback.format_exc())

# ── Test 2: open board — action decreases row (moves toward goal row 0) ───────
try:
    state = State()
    N = state.N
    start_row = state.player[0] // N
    action = bfs_forward_action(state)
    new_row = action // N
    check("open board: moves toward row 0", new_row < start_row,
          f"start_row={start_row}, new_row={new_row}")
except Exception as e:
    check("open board row advance", False, traceback.format_exc())

# ── Test 3: already at goal row — returns some legal action without crash ─────
try:
    state = State()
    N = state.N
    # Manually teleport player to row 0 (position 0..N-1)
    state.player[0] = 0  # top-left corner, row 0
    action = bfs_forward_action(state)
    check("already at goal: no crash", True)
    check("already at goal: action is legal", action in state.legal_actions(),
          f"action={action}")
except Exception as e:
    check("already at goal", False, traceback.format_exc())

# ── Test 4: opponent adjacent — jump handled correctly ───────────────────────
try:
    state = State()
    N = state.N
    # Put player at row 2, enemy at row 1 (directly in the way)
    state.player[0] = 2 * N + N // 2   # row 2, middle col
    state.enemy[0]  = 1 * N + N // 2   # row 1, same col (mirror position)
    state._legal_actions_cache = None
    action = bfs_forward_action(state)
    legal = state.legal_actions()
    pos_legal = [a for a in legal if a < N * N]
    check("jump scenario: action is legal pawn move", action in pos_legal,
          f"action={action}, pos_legal={pos_legal}")
    # Should jump over to row 0 or go diagonal — either way row must be < 2
    result_row = action // N
    check("jump scenario: advances past enemy", result_row < 2,
          f"result_row={result_row}")
except Exception as e:
    check("jump scenario", False, traceback.format_exc())

# ── Test 5: BFS vs BFS full game — no crash, terminates ──────────────────────
try:
    state = State()
    moves = 0
    while not state.is_done():
        if state.is_first_player():
            action = bfs_forward_action(state)
        else:
            # Enemy is mirrored — use greedy as stand-in (BFS needs player perspective)
            action = greedy_forward_action(state)
        assert action in state.legal_actions(), f"Illegal action {action} at move {moves}"
        state = state.next(action)
        moves += 1
        assert moves < 500, "Game did not terminate (>500 moves)"
    check("full game BFS vs greedy: terminates", True, f"{moves} moves")
    check("full game BFS vs greedy: game is done", state.is_done())
except Exception as e:
    check("full game BFS vs greedy", False, traceback.format_exc())

# ── Test 6: 20 random games BFS as player 1 — no crash ───────────────────────
try:
    import random
    from game import random_action
    crashes = 0
    for game_i in range(20):
        state = State()
        moves = 0
        try:
            while not state.is_done():
                if state.is_first_player():
                    action = bfs_forward_action(state)
                else:
                    action = random_action(state)
                state = state.next(action)
                moves += 1
                if moves > 500:
                    raise RuntimeError("did not terminate")
        except Exception as inner:
            crashes += 1
            print(f"  game {game_i} crashed: {inner}")
    check("20 games BFS vs random: no crashes", crashes == 0,
          f"{crashes} crashes")
except Exception as e:
    check("20 games BFS vs random", False, traceback.format_exc())

# ── Test 7: 20 random games BFS as player 2 — no crash ───────────────────────
try:
    crashes = 0
    for game_i in range(20):
        state = State()
        moves = 0
        try:
            while not state.is_done():
                if state.is_first_player():
                    action = random_action(state)
                else:
                    action = bfs_forward_action(state)
                state = state.next(action)
                moves += 1
                if moves > 500:
                    raise RuntimeError("did not terminate")
        except Exception as inner:
            crashes += 1
            print(f"  game {game_i} crashed: {inner}")
    check("20 games random vs BFS: no crashes", crashes == 0,
          f"{crashes} crashes")
except Exception as e:
    check("20 games random vs BFS", False, traceback.format_exc())

# ── Summary ───────────────────────────────────────────────────────────────────
print()
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"Results: {passed} passed, {failed} failed")
if failed:
    print("FAILURES:")
    for s, name, detail in results:
        if s == FAIL:
            print(f"  {name}: {detail}")
    sys.exit(1)
