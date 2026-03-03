"""Quick visual test for _bfs_goal_distances.

Run with:  python test_bfs_distances.py
"""
from game import _get_blocked_edges, _bfs_goal_distances

N = 7

def print_grid(title, dist, highlight_pos=None):
    print(f'\n{title}')
    print('    ' + '  '.join(f'c{c}' for c in range(N)))
    for r in range(N):
        row_vals = []
        for c in range(N):
            pos = r * N + c
            val = dist[pos]
            marker = f'[{val:2d}]' if pos == highlight_pos else f' {val:2d} '
            row_vals.append(marker)
        label = 'GOAL' if r == 0 else f'r{r}  '
        print(f'{label}  ' + ' '.join(row_vals))


def make_walls(*placements):
    """Convenience: list of (wp_index, orientation) where orientation=1=H, 2=V."""
    walls = [0] * ((N - 1) ** 2)
    for wp, ori in placements:
        walls[wp] = ori
    return tuple(walls)


# ── Scenario 1: empty board ────────────────────────────────────────────────────
walls = make_walls()
h, v = _get_blocked_edges(N, walls)
dist0 = _bfs_goal_distances(N, h, v, 0)
distN = _bfs_goal_distances(N, h, v, N - 1)
print_grid('Empty board — distance to row 0 (player goal)', dist0, highlight_pos=6*N+3)
print_grid('Empty board — distance to row 6 (opponent goal)', distN, highlight_pos=3)


# ── Scenario 2: horizontal wall blocking column 3 between rows 3 and 4 ────────
# Wall slot index for row=3, col=3 → wp = 3*(N-1) + 3 = 3*6+3 = 21
# This is a horizontal wall: blocks up-movement at (r4,c3) and (r4,c4)
walls = make_walls((21, 1), (22, 1))   # two adjacent H-walls making a longer barrier
h, v = _get_blocked_edges(N, walls)
dist0 = _bfs_goal_distances(N, h, v, 0)
print_grid('Two H-walls at slots 21,22 (across cols 3-5 between rows 3-4) — dist to row 0', dist0, highlight_pos=4*N+3)


# ── Scenario 3: vertical wall corridor on the left ────────────────────────────
# Vertical walls at col 1 running down rows 0-1 and 2-3
# wp for (r=0,c=0) = 0, orientation=2 (vertical) → blocks left/right at col 1 rows 0-1
# wp for (r=2,c=0) = 2*(N-1) + 0 = 12
walls = make_walls((0, 2), (6, 2), (12, 2), (18, 2), (24, 2))  # full vertical wall col 1 rows 0-5
h, v = _get_blocked_edges(N, walls)
dist0 = _bfs_goal_distances(N, h, v, 0)
print_grid('Full V-wall sealing off column 0 — dist to row 0\n  (col 0 forced to detour around top)', dist0, highlight_pos=6*N+0)


# ── Scenario 4: pawn at various positions, show their actual BFS distance ─────
print('\n\nSpot-check: specific pawn positions on empty board')
walls = make_walls()
h, v = _get_blocked_edges(N, walls)
dist0 = _bfs_goal_distances(N, h, v, 0)
for (r, c) in [(6, 3), (3, 3), (1, 3), (0, 3), (6, 0), (6, 6)]:
    pos = r * N + c
    print(f'  Pawn at (r={r}, c={c}) → {dist0[pos]} moves to row 0')
