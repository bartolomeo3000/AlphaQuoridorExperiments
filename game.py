# ====================
# Quoridor (3 x 3), wall = 1
# ====================

# Importing packages
import random
import math
from collections import deque
import copy
from copy import deepcopy

from config import DRAW_DEPTH

# Global cache for expensive wall legality computation.
# Key: (walls_tuple, player_pos, enemy_pos)  Value: list of legal wall action indices
# Shared across all State instances — MCTS nodes with the same board config pay the BFS cost only once.
_wall_actions_cache = {}

# Edge-blocking cache: walls_tuple -> (h_blocked, v_blocked) bytearrays.
# h_blocked[pos] = 1  means the edge pos → pos-N (moving UP) is blocked by a wall.
# v_blocked[pos] = 1  means the edge pos → pos-1 (moving LEFT) is blocked by a wall.
_edge_cache = {}
_bfs_dist_cache = {}  # walls_tuple -> (dist_to_row0, dist_to_rowN-1)


def _get_blocked_edges(N, walls_tuple):
    """Build (and cache) edge-blocking tables for a given wall configuration.
    Horizontal wall at (wr,wc) blocks vertical movement at (wr+1,wc) and (wr+1,wc+1).
    Vertical   wall at (wr,wc) blocks horizontal movement at (wr,wc+1) and (wr+1,wc+1).
    """
    if walls_tuple in _edge_cache:
        return _edge_cache[walls_tuple]
    h = bytearray(N * N)
    v = bytearray(N * N)
    w1 = N - 1
    for wp, w in enumerate(walls_tuple):
        if not w:
            continue
        wr, wc = wp // w1, wp % w1
        if w == 1:  # horizontal
            h[(wr + 1) * N + wc]     = 1
            h[(wr + 1) * N + wc + 1] = 1
        else:       # vertical
            v[wr       * N + wc + 1] = 1
            v[(wr + 1) * N + wc + 1] = 1
    result = (h, v)
    if len(_edge_cache) < 200_000:
        _edge_cache[walls_tuple] = result
    return result


def _bfs_goal_distances(N, h, v, goal_row):
    """Multi-source BFS returning the distance (in moves) from every cell to the
    nearest cell in goal_row, using the same h/v edge arrays as _bfs_can_reach_row0.
    Returns a flat list of N² ints. Cells cut off by walls get distance N*N (sentinel).
    """
    INF = N * N
    dist = [INF] * (N * N)
    queue = deque()
    for c in range(N):
        pos = goal_row * N + c
        dist[pos] = 0
        queue.append(pos)
    while queue:
        pos = queue.popleft()
        r   = pos // N
        c_  = pos %  N
        d   = dist[pos]
        # Up
        if r > 0 and not h[pos]:
            nb = pos - N
            if dist[nb] == INF:
                dist[nb] = d + 1
                queue.append(nb)
        # Down
        if r < N - 1 and not h[pos + N]:
            nb = pos + N
            if dist[nb] == INF:
                dist[nb] = d + 1
                queue.append(nb)
        # Left
        if c_ > 0 and not v[pos]:
            nb = pos - 1
            if dist[nb] == INF:
                dist[nb] = d + 1
                queue.append(nb)
        # Right
        if c_ < N - 1 and not v[pos + 1]:
            nb = pos + 1
            if dist[nb] == INF:
                dist[nb] = d + 1
                queue.append(nb)
    return dist


def _bfs_can_reach_row0(N, h, v, start):
    """Return True if 'start' can reach row 0 (top row) ignoring piece positions."""
    if start // N == 0:
        return True
    visited = bytearray(N * N)
    visited[start] = 1
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        r = pos // N
        c = pos % N
        # Up
        if r > 0 and not h[pos]:
            nb = pos - N
            if not visited[nb]:
                if nb // N == 0:
                    return True
                visited[nb] = 1
                queue.append(nb)
        # Down
        if r < N - 1 and not h[pos + N]:
            nb = pos + N
            if not visited[nb]:
                if nb // N == 0:
                    return True
                visited[nb] = 1
                queue.append(nb)
        # Left
        if c > 0 and not v[pos]:
            nb = pos - 1
            if not visited[nb]:
                if nb // N == 0:
                    return True
                visited[nb] = 1
                queue.append(nb)
        # Right
        if c < N - 1 and not v[pos + 1]:
            nb = pos + 1
            if not visited[nb]:
                if nb // N == 0:
                    return True
                visited[nb] = 1
                queue.append(nb)
    return False

# Game state
class State:
    def __init__(self, board_size=7, num_walls=5, player=None, enemy=None, walls=None, depth=0, pos_counts=None):
        self.N = board_size
        N = self.N
        if N % 2 == 0:
            raise ValueError('The board size must be an odd number.')
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.player = player if player != None else [0] * 2 # Position, number of walls
        self.enemy = enemy if enemy != None else [0] * 2
        self.walls = walls if walls != None else [0] * ((N - 1) ** 2)
        self.depth = depth
        self._legal_actions_cache = None  # computed once on first access
        self._pos_counts = dict(pos_counts) if pos_counts is not None else {}
        self._repetition_draw = False

        if player == None or enemy == None:
            init_pos = N * (N - 1) + N // 2
            self.player[0] = init_pos
            self.player[1] = num_walls
            self.enemy[0] = init_pos
            self.enemy[1] = num_walls
            self._register_position()  # register starting position

    def _register_position(self):
        """Record current position; set _repetition_draw if seen 3 times."""
        key = (self.player[0], self.enemy[0], tuple(self.walls))
        count = self._pos_counts.get(key, 0) + 1
        self._pos_counts[key] = count
        if count >= 3:
            self._repetition_draw = True

     # Check if it's a loss
    def is_lose(self):
        if self.enemy[0] // self.N == 0:
            return True
        return False

    # Check if it's a draw
    def is_draw(self):
        return self.depth >= DRAW_DEPTH or self._repetition_draw
    
    # Check if the game is over
    def is_done(self):
        return self.is_lose() or self.is_draw()

    def bfs_distances(self):
        """Return (p_dist, e_dist): wall-aware BFS move-count to goal for each player.

        p_dist — moves for current player to reach row 0 (their goal), using current walls.
        e_dist — moves for enemy to reach row 0 (their goal), using ROTATED walls.

        Walls must be rotated for the enemy because next() calls rotate_walls() before
        swapping players — a wall near row 0 in the current frame is near row N-1 in
        the enemy's frame (near their start, not their goal).  This mirrors exactly what
        legal_actions_wall uses for enemy reachability checks.

        Reuses _bfs_dist_cache populated by pieces_array(); computes on demand
        if the entry isn't cached yet (e.g. called before pieces_array).
        """
        N = self.N
        walls_t = tuple(self.walls)
        walls_rot = tuple(reversed(walls_t))

        # Current player: dist0 with current walls
        if walls_t in _bfs_dist_cache:
            dist0_curr = _bfs_dist_cache[walls_t][0]
        else:
            h, v = _get_blocked_edges(N, walls_t)
            dist0_curr = _bfs_goal_distances(N, h, v, 0)
            distN1     = _bfs_goal_distances(N, h, v, N - 1)
            if len(_bfs_dist_cache) < 200_000:
                _bfs_dist_cache[walls_t] = (dist0_curr, distN1)

        # Enemy: dist0 with rotated walls (their perspective after next()'s rotate_walls)
        if walls_rot in _bfs_dist_cache:
            dist0_rot = _bfs_dist_cache[walls_rot][0]
        else:
            h_r, v_r = _get_blocked_edges(N, walls_rot)
            dist0_rot  = _bfs_goal_distances(N, h_r, v_r, 0)
            distN1_rot = _bfs_goal_distances(N, h_r, v_r, N - 1)
            if len(_bfs_dist_cache) < 200_000:
                _bfs_dist_cache[walls_rot] = (dist0_rot, distN1_rot)

        return dist0_curr[self.player[0]], dist0_rot[self.enemy[0]]

    def pieces_array(self):
        N = self.N
        def pieces_of(pieces, flip=False):
            tables = []

            table = [0] * (N ** 2)
            pos = N ** 2 - 1 - pieces[0] if flip else pieces[0]
            table[pos] = 1
            tables.append(table)
                
            table = [pieces[1]] * (N ** 2)
            tables.append(table)

            return tables
        
        def walls_of(walls):
            # Use the engine's edge-blocking representation directly:
            #   h[pos] = 1  →  moving UP   from cell pos is blocked
            #   v[pos] = 1  →  moving LEFT from cell pos is blocked
            # Each wall correctly marks the two cells whose edges it blocks,
            # matching exactly what the movement/BFS logic uses.
            h, v = _get_blocked_edges(N, tuple(walls))
            return [list(h), list(v)]
        
        result = [pieces_of(self.player), pieces_of(self.enemy, flip=True), walls_of(self.walls)]

        from config import USE_BFS_CHANNELS
        if USE_BFS_CHANNELS:
            walls_t = tuple(self.walls)
            if walls_t not in _bfs_dist_cache:
                h, v = _get_blocked_edges(N, walls_t)
                entry = (
                    _bfs_goal_distances(N, h, v, 0),      # ch7: dist to row 0  (current player's goal)
                    _bfs_goal_distances(N, h, v, N - 1),  # ch8: dist to row N-1 (same walls, opposite direction)
                )
                if len(_bfs_dist_cache) < 200_000:
                    _bfs_dist_cache[walls_t] = entry
            else:
                entry = _bfs_dist_cache[walls_t]
            result.append(list(entry))

        return result

    def legal_actions(self):
        """
        0 - (N ** 2 - 1): Move to a position
        N ** 2- (N ** 2 + (N - 1) ** 2 - 1): Place a horizontal wall
        (N ** 2 + (N - 1) ** 2) - (N ** 2 + 2 * (N - 1) ** 2 - 1): Place a vertical wall
        """
        if self._legal_actions_cache is not None:
            return self._legal_actions_cache

        actions = list(self.legal_actions_pos(self.player[0]))

        if self.player[1] > 0:
            cache_key = (tuple(self.walls), self.player[0], self.enemy[0])
            if cache_key in _wall_actions_cache:
                actions.extend(_wall_actions_cache[cache_key])
            else:
                wall_actions = []
                for pos in range((self.N - 1) ** 2):
                    wall_actions.extend(self.legal_actions_wall(pos))
                if len(_wall_actions_cache) < 300_000:
                    _wall_actions_cache[cache_key] = wall_actions
                actions.extend(wall_actions)

        self._legal_actions_cache = actions
        return actions

    def legal_actions_pos(self, pos):
        actions = []

        N = self.N
        walls = self.walls
        ep = self.enemy[0]

        x, y = pos // N, pos % N
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                np = N * nx + ny
                wp = (N - 1) * nx + ny

                if nx < x:
                    if y == 0:
                        if walls[wp] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx > 0 and walls[wp - (N - 1)] != 1:
                                    nnp = np - N
                                    actions.append(nnp)
                                elif (nx == 0 and walls[wp] != 2) or (nx > 0 and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
                                    nnp = np + 1
                                    actions.append(nnp)
                    elif y == (N - 1):
                        if walls[wp - 1] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx > 0 and walls[wp - (N - 1) - 1] != 1:
                                    nnp = np -  N
                                    actions.append(nnp)
                                elif (nx == 0 and walls[wp - 1] != 2) or (nx > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                    nnp = np - 1
                                    actions.append(nnp)
                    else:
                        if walls[wp - 1] != 1 and walls[wp] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx > 0 and walls[wp - (N - 1)] != 1 and walls[wp - (N - 1) - 1] != 1:
                                    nnp = np - N
                                    actions.append(nnp)
                                else:
                                    if (nx == 0 and walls[wp - 1] != 2) or (nx > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                        nnp = np - 1
                                        actions.append(nnp)
                                    if (nx == 0 and walls[wp] != 2) or (nx > 0 and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
                                        nnp = np + 1
                                        actions.append(nnp)
                if nx > x:
                    if y == 0:
                        if walls[wp - (N - 1)] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx < (N - 1) and walls[wp] != 1:
                                    nnp = np + N
                                    actions.append(nnp)
                                elif (nx == (N - 1) and walls[wp - (N - 1)] != 2) or (nx < (N - 1) and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
                                    nnp = np + 1
                                    actions.append(nnp)
                    elif y == (N - 1):
                        if walls[wp - (N - 1) - 1] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx < (N - 1) and walls[wp - 1] != 1:
                                    nnp = np + N
                                    actions.append(nnp)
                                elif (nx == (N - 1) and walls[wp - (N - 1) - 1] != 2) or (nx < (N - 1) and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                    nnp = np - 1
                                    actions.append(nnp)
                    else:
                        if walls[wp - (N - 1) - 1] != 1 and walls[wp - (N - 1)] != 1:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if nx < (N - 1) and walls[wp - 1] != 1 and walls[wp] != 1:
                                    nnp = np + N
                                    actions.append(nnp)
                                else:
                                    if (nx == (N - 1) and walls[wp - (N - 1) - 1] != 2) or (nx < (N - 1) and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2):
                                        nnp = np - 1
                                        actions.append(nnp)
                                    if (nx == (N - 1) and walls[wp - (N - 1)] != 2) or (nx < (N - 1) and walls[wp - (N - 1)] != 2 and walls[wp] != 2):
                                        nnp = np + 1
                                        actions.append(nnp)
                if ny < y:
                    if x == 0:
                        if walls[wp] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny > 0 and walls[wp - 1] != 2:
                                    nnp = np - 1
                                    actions.append(nnp)
                                elif (ny == 0 and walls[wp] != 1) or (ny > 0 and walls[wp - 1] != 1 and walls[wp] != 1):
                                    nnp = np + N
                                    actions.append(nnp)
                    elif x == (N - 1):
                        if walls[wp - (N - 1)] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny > 0 and walls[wp - (N - 1) - 1] != 2:
                                    nnp = np - 1
                                    actions.append(nnp)
                                elif (ny == 0 and walls[wp - (N - 1)] != 1) or (ny > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - (N - 1)] != 1):
                                    nnp = np - N
                                    actions.append(nnp)
                    else:
                        if walls[wp - (N - 1)] != 2 and walls[wp] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2:
                                    nnp = np - 1
                                    actions.append(nnp)
                                else:
                                    if (ny == 0 and walls[wp - (N - 1)] != 1) or (ny > 0 and walls[wp - (N - 1) - 1] != 2 and walls[wp - (N - 1)] != 1):
                                        nnp = np - N
                                        actions.append(nnp)
                                    if (ny == 0 and walls[wp] != 1) or (ny > 0 and (walls[wp - 1] != 1 or walls[wp] != 1)):
                                        nnp = np + N
                                        actions.append(nnp)
                if ny > y:
                    if x == 0:
                        if walls[wp - 1] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny < (N - 1) and walls[wp] != 2:
                                    nnp = np + 1
                                    actions.append(nnp)
                                elif (ny == (N - 1) and walls[wp - 1] != 1) or (ny < (N - 1) and walls[wp - 1] != 1 and walls[wp] != 1):
                                    nnp = np + N
                                    actions.append(nnp)
                    elif x == (N - 1):
                        if walls[wp - (N - 1) - 1] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny < (N - 1) and walls[wp - (N - 1)] != 2:
                                    nnp = np + 1
                                    actions.append(nnp)
                                elif (ny == (N - 1) and walls[wp - (N - 1) - 1] != 1) or (ny < (N - 1) and walls[wp - (N - 1) - 1] != 1 and walls[wp - (N - 1)] != 1):
                                    nnp = np - N
                                    actions.append(nnp)
                    else:
                        if walls[wp - (N - 1) - 1] != 2 and walls[wp - 1] != 2:
                            if np + ep != N ** 2 - 1:
                                actions.append(np)
                            else:
                                if ny < (N - 1) and walls[wp - (N - 1)] != 2 and walls[wp] != 2:
                                    nnp = np + 1
                                    actions.append(nnp)
                                else:
                                    if (ny == (N - 1) and walls[wp - (N - 1) - 1] != 1) or (ny < (N - 1) and walls[wp - (N - 1) - 1] != 1 and walls[wp - (N - 1)] != 1):
                                        nnp = np - N
                                        actions.append(nnp)
                                    if (ny == (N - 1) and walls[wp - 1] != 1) or (ny < (N - 1) and (walls[wp - 1] != 1 or walls[wp] != 1)):
                                        nnp = np + N
                                        actions.append(nnp)

        return actions

    def legal_actions_wall(self, pos):
        N = self.N
        walls = self.walls
        def can_place_wall(orientation, pos):
            if walls[pos] != 0:
                return False
            x, y = pos // (N - 1), pos % (N - 1)
            if orientation == 1:
                if y == 0:
                    if walls[pos + 1] == 1:
                        return False
                elif y == (N - 2):
                    if walls[pos - 1] == 1:
                        return False
                else:
                    if walls[pos - 1] == 1 or walls[pos + 1] == 1:
                        return False
            else:
                if x == 0:
                    if walls[pos + (N - 1)] == 2:
                        return False
                elif x == (N - 2):
                    if walls[pos - (N - 1)] == 2:
                        return False
                else:
                    if walls[pos - (N - 1)] == 2 or walls[pos + (N - 1)] == 2:
                        return False
            return True

        def can_reach_goal(orientation, pos):
            # Temporarily place wall, snapshot as tuple, then restore
            self.walls[pos] = orientation
            trial = tuple(self.walls)
            self.walls[pos] = 0

            # Player reachability with trial walls
            h, v = _get_blocked_edges(N, trial)
            if not _bfs_can_reach_row0(N, h, v, self.player[0]):
                return False

            # Enemy reachability with rotated trial walls (mirrors next() rotate_walls)
            h_r, v_r = _get_blocked_edges(N, tuple(reversed(trial)))
            return _bfs_can_reach_row0(N, h_r, v_r, self.enemy[0])
    
        actions = []

        if can_place_wall(1, pos) and can_reach_goal(1, pos):
            actions.append(N ** 2 + pos)
        if can_place_wall(2, pos) and can_reach_goal(2, pos):
            actions.append(N ** 2 + (N - 1) ** 2 + pos)

        return actions
    
    def rotate_walls(self):
        N = self.N
        rotated_walls = [0] * len(self.walls)
        for i in range((N - 1) ** 2):
            rotated_walls[i] = self.walls[(N - 1) ** 2 - 1 - i]
        self.walls = rotated_walls
    
    def next(self, action):
        N = self.N
        # Create the next state, passing position history
        state = State(board_size=N, player=self.player.copy(), enemy=self.enemy.copy(), walls=deepcopy(self.walls), depth=self.depth + 1, pos_counts=self._pos_counts)

        if action < N ** 2:
            # Move piece
            state.player[0] = action
        elif action < N ** 2 + (N - 1) ** 2:
            # Place horizontal wall
            pos = action - N ** 2
            state.walls[pos] = 1
            state.player[1] -= 1
        else:
            # Place vertical wall
            pos = action - N ** 2 - (N - 1) ** 2
            state.walls[pos] = 2
            state.player[1] -= 1

        state.rotate_walls()

        # Swap players
        state.player, state.enemy = state.enemy, state.player

        # Register position after all mutations are done
        state._register_position()

        return state
    
    # Check if it's the first player's turn
    def is_first_player(self):
        return self.depth % 2 == 0

    def __str__(self):
        """Display the game state as a string."""
        N = self.N
        is_first_player = self.is_first_player()

        board = [['o'] * (2 * N - 1) for _ in range(2 * N - 1)]
        for i in range(2 * N - 1):
            for j in range(2 * N - 1):
                if i % 2 == 1 and j % 2 == 1:
                    board[i][j] = 'x'

        p_pos = self.player[0] if is_first_player else self.enemy[0]
        e_pos = self.enemy[0] if is_first_player else self.player[0]

        e_pos = N ** 2 - 1 - e_pos

        p_x, p_y = p_pos // N, p_pos % N
        e_x, e_y = e_pos // N, e_pos % N

        board[2 * p_x][2 * p_y] = 'P'
        board[2 * e_x][2 * e_y] = 'E'
        
        turn_info = "<Enemy's Turn>" if is_first_player else "<Player's Turn>"

        if not is_first_player:
            self.rotate_walls()

        # Set walls
        for i in range(N - 1):
            for j in range(N - 1):
                pos = i * (N - 1) + j
                if self.walls[pos] == 1:
                    board[2 * i + 1][2 * j] = '-'
                    board[2 * i + 1][2 * (j + 1)] = '-'
                if self.walls[pos] == 2:
                    board[2 * i][2 * j + 1] = '|'
                    board[2 * (i + 1)][2 * j + 1] = '|'

        if not is_first_player:
            self.rotate_walls()

        board_str = '\n'.join([''.join(row) for row in board])
        return turn_info + '\n' + board_str


# Randomly select an action
def random_action(state):
    legal_actions = state.legal_actions()
    action = random.randint(0, len(legal_actions) - 1)
    return legal_actions[action]

# Calculate state value using alpha-beta pruning
def alpha_beta(state, alpha, beta):
    # Loss is -1
    if state.is_lose():
        return -1

    # Draw is 0
    if state.is_draw():
        return 0

    # Calculate state values for legal actions
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score

        # If the best score for the current node exceeds the parent node, stop the search
        if alpha >= beta:
            return alpha

    # Return the maximum value of the state values for legal actions
    return alpha

# Select an action using alpha-beta pruning
def alpha_beta_action(state):
    # Calculate state values for legal actions
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score

    # Return the action with the maximum state value
    return best_action

# Playout
def playout(state):
    # Loss is -1
    if state.is_lose():
        return -1

    # Draw is 0
    if state.is_draw():
        return 0

    # Next state value
    return -playout(state.next(random_action(state)))

# Return the index of the maximum value
def argmax(collection):
    return collection.index(max(collection))

# Select an action using Monte Carlo Tree Search
def mcts_action(state):
    # Node for Monte Carlo Tree Search
    class node:
        # Initialization
        def __init__(self, state):
            self.state = state  # State
            self.w = 0  # Cumulative value
            self.n = 0  # Number of trials
            self.child_nodes = None  # Child nodes

        # Evaluation
        def evaluate(self):
            # When the game ends
            if self.state.is_done():
                # Get value from the game result
                value = -1 if self.state.is_lose() else 0  # Loss is -1, draw is 0

                # Update cumulative value and number of trials
                self.w += value
                self.n += 1
                return value

            # When there are no child nodes
            if not self.child_nodes:
                # Get value from playout
                value = playout(self.state)

                # Update cumulative value and number of trials
                self.w += value
                self.n += 1

                # Expand child nodes
                if self.n == 10:
                    self.expand()
                return value

            # When there are child nodes
            else:
                # Get value from evaluating the child node with the maximum UCB1
                value = -self.next_child_node().evaluate()

                # Update cumulative value and number of trials
                self.w += value
                self.n += 1
                return value

        # Expand child nodes
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(node(self.state.next(action)))

        # Get the child node with the maximum UCB1
        def next_child_node(self):
            # Return the child node with n=0
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # Calculate UCB1
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w/child_node.n + 2*(2*math.log(t)/child_node.n)**0.5)

            # Return the child node with the maximum UCB1
            return self.child_nodes[argmax(ucb1_values)]

    # Generate the root node
    root_node = node(state)
    root_node.expand()

    # Evaluate the root node 100 times
    for _ in range(100):
        root_node.evaluate()

    # Return the action with the maximum number of trials
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

# Running the function
if __name__ == '__main__':
    # Generate the state
    state = State()

    # Loop until the game ends
    while True:
        # When the game ends
        if state.is_done():
            break

        # Get the next state
        state = state.next(random_action(state))

        # Display as a string
        print(state)
        print()
