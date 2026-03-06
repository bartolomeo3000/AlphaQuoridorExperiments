# Debug version of human_play.
# After every move, prints the exact 8-channel input the network receives,
# plus pawn positions, to the terminal.

import os
import numpy as np
from game import State
from pv_mcts import pv_mcts_action, random_action
from dual_network import load_model, DN_INPUT_SHAPE
from config import MODEL_DIR, USE_BFS_CHANNELS
import tkinter as tk
import pv_mcts
pv_mcts.PV_EVALUATE_COUNT = 200  # keep fast for debug sessions

model = load_model(os.path.join(MODEL_DIR, 'best.pt'))

# ── Debug printer ─────────────────────────────────────────────────────────────

CH_NAMES = [
    'ch1  player position  (1 = pawn)',
    'ch2  player walls     (remaining count, uniform)',
    'ch3  enemy  position  (1 = pawn, stored pre-mirrored)',
    'ch4  enemy  walls     (remaining count, uniform)',
    'ch5  H-wall map       (1 = horizontal wall edge)',
    'ch6  V-wall map       (1 = vertical   wall edge)',
    'ch7  BFS dist→row0    (current player goal, current walls)',
    'ch8  BFS dist→rowN-1  (opposite direction, current walls)',
]

def print_nn_input(state, move_label=''):
    N = state.N
    pa = state.pieces_array()
    # pieces_array() returns nested lists; flatten to (C, N, N)
    x = np.array(pa, dtype=np.float32).reshape(-1, N, N)
    C = x.shape[0]

    pp = state.player[0]
    ep = state.enemy[0]

    print()
    print(f'{"─"*60}')
    print(f'  NN INPUT  {move_label}')
    print(f'  player stored pos = {pp}  (row {pp//N}, col {pp%N})')
    print(f'  enemy  stored pos = {ep}  (row {ep//N}, col {ep%N})'
          f'  [visual row {(N*N-1-ep)//N}, col {(N*N-1-ep)%N}]')

    # BFS distances at actual pawn positions
    if USE_BFS_CHANNELS:
        p_dist, e_dist = state.bfs_distances()
        print(f'  BFS dist player→goal = {p_dist}   enemy→goal = {e_dist}')

    for c in range(C):
        name = CH_NAMES[c] if c < len(CH_NAMES) else f'ch{c+1}'
        print(f'\n  [{c+1}] {name}')
        grid = x[c]
        # column header
        print('      ' + ' '.join(f'{j:4}' for j in range(N)))
        print('      ' + '─'*5*N)
        for r in range(N):
            vals = ' '.join(f'{grid[r,j]:4.0f}' for j in range(N))
            goal_marker = ' ← GOAL' if r == 0 and c >= 6 else ''
            start_marker = ' ← START' if r == N-1 and c >= 6 else ''
            print(f' r{r} |  {vals}{goal_marker}{start_marker}')
    print(f'{"─"*60}')
    print()


# ── UI (identical to human_play) ──────────────────────────────────────────────

class GameUI(tk.Frame):
    def __init__(self, master=None, model=None):
        tk.Frame.__init__(self, master)
        self.master.title('Quoridor [DEBUG]')

        self.state = State()
        self.N = self.state.N
        self.D = 80

        self.select = -1
        self.placing_wall = False

        self.next_action = pv_mcts_action(model) if model else random_action()

        self.grid(sticky='nsew')
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.c = tk.Canvas(self, highlightthickness=0, bg='#4B4B4B')
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.bind('<Configure>', self._on_resize)
        self.c.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        self.player_walls_frame = tk.Frame(self)
        self.player_walls_frame.grid(row=1, column=2, padx=10, pady=10)
        self.player_walls = tk.Label(self.player_walls_frame, text="Player Walls",
                                     anchor="center", justify=tk.CENTER, font=('Helvetica', 14))
        self.player_walls.pack()

        self.enemy_walls_frame = tk.Frame(self)
        self.enemy_walls_frame.grid(row=1, column=0, padx=10, pady=10)
        self.enemy_walls = tk.Label(self.enemy_walls_frame, text="Enemy Walls",
                                    anchor="center", justify=tk.CENTER, font=('Helvetica', 14))
        self.enemy_walls.pack()

        self.controls_frame = tk.Frame(self)
        self.controls_frame.grid(row=2, column=1, padx=10, pady=10)
        self.wall_button = tk.Button(self.controls_frame, text="Place Wall",
                                     command=self.place_wall_mode)
        self.wall_button.pack()

        self.wall_direction = tk.StringVar(value="horizontal")
        tk.Radiobutton(self.controls_frame, text="Horizontal",
                       variable=self.wall_direction, value="horizontal").pack()
        tk.Radiobutton(self.controls_frame, text="Vertical",
                       variable=self.wall_direction, value="vertical").pack()

        self.result_message = tk.Label(self, text="", font=('Helvetica', 24))
        self.result_message.grid(row=0, column=1, pady=10)

        screen_w = master.winfo_screenwidth()
        screen_h = master.winfo_screenheight()
        init = min(int(screen_w * 0.6), int(screen_h * 0.8))
        master.geometry(f'{init}x{int(init * 0.85)}')

        # Print initial state
        print_nn_input(self.state, move_label='initial state (human to move)')
        self.on_draw()

    def _on_resize(self, event):
        board_px = min(event.width, event.height)
        self.D = max(20, board_px // self.N)
        self.on_draw()

    def place_wall_mode(self):
        self.placing_wall = not self.placing_wall
        self.wall_button.config(text="Move Piece" if self.placing_wall else "Place Wall")

    def turn_of_human(self, event):
        N = self.N
        D = self.D
        if self.state.is_done():
            return
        if not self.state.is_first_player():
            return

        if self.placing_wall:
            x, y = (event.x - D // 2) // D, (event.y - D // 2) // D
            if 0 <= x < N - 1 and 0 <= y < N - 1:
                if not self.place_wall(x, y):
                    return
            else:
                return
        else:
            x, y = event.x // D, event.y // D
            action = N * y + x
            if action not in self.state.legal_actions():
                self.select = -1
                self.on_draw()
                return
            self.state = self.state.next(action)
            self.select = -1
            self.on_draw()
            # Print the state the AI now receives
            print_nn_input(self.state, move_label='after human move (AI to move)')

        self.master.after(500, self.turn_of_ai)

    def place_wall(self, x, y):
        N = self.N
        if self.wall_direction.get() == "horizontal":
            action = N ** 2 + (N - 1) * y + x
        else:
            action = N ** 2 + (N - 1) ** 2 + (N - 1) * y + x

        if action in self.state.legal_actions():
            self.state = self.state.next(action)
            self.placing_wall = False
            self.wall_button.config(text="Place Wall")
            self.on_draw()
            print_nn_input(self.state, move_label='after human wall (AI to move)')
            return True
        else:
            self.on_draw()
            return False

    def turn_of_ai(self):
        if self.state.is_done():
            self.display_result()
            self.master.after(1000, self.reset_game)
            return

        action = self.next_action(self.state)
        self.state = self.state.next(action)
        self.on_draw()
        # Print the state the human now receives
        print_nn_input(self.state, move_label='after AI move (human to move)')

        if self.state.is_done():
            self.display_result()
            self.master.after(1000, self.reset_game)

    def display_result(self):
        if self.state.is_draw():
            self.result_message.config(text="Draw", fg="gray")
            return
        is_lose = self.state.is_lose() if self.state.is_first_player() else not self.state.is_lose()
        self.result_message.config(
            text="You Lose" if is_lose else "You Win",
            fg="blue" if is_lose else "red"
        )

    def reset_game(self):
        self.state = State()
        print_nn_input(self.state, move_label='NEW GAME — initial state (human to move)')
        self.on_draw()
        self.result_message.config(text="")

    def draw_piece(self, index, color):
        N, D = self.N, self.D
        x = (index % N) * D
        y = (index // N) * D
        margin = D // 10
        self.c.create_oval(x + margin, y + margin, x + D - margin, y + D - margin,
                           fill=color, outline='black')

    def draw_walls(self):
        N, D = self.N, self.D
        for i in range(len(self.state.walls)):
            x, y = i % (N - 1), i // (N - 1)
            if self.state.walls[i] == 1:
                self.c.create_line(x * D, (y + 1) * D, (x + 2) * D, (y + 1) * D,
                                   width=max(4, D // 12), fill='#D1B575')
            elif self.state.walls[i] == 2:
                self.c.create_line((x + 1) * D, y * D, (x + 1) * D, (y + 2) * D,
                                   width=max(4, D // 12), fill='#D1B575')

    def on_draw(self):
        N, D, L = self.N, self.D, self.N * self.D
        is_fp = self.state.is_first_player()
        self.c.delete('all')
        self.c.create_rectangle(0, 0, L, L, width=0.0, fill='#4B4B4B')
        for i in range(1, N):
            self.c.create_line(i * D, 0, i * D, L, width=max(4, D // 12), fill='#8B0000')
            self.c.create_line(0, i * D, L, i * D, width=max(4, D // 12), fill='#8B0000')

        p_pos = self.state.player[0] if is_fp else self.state.enemy[0]
        e_pos = self.state.enemy[0]  if is_fp else self.state.player[0]
        e_pos = N ** 2 - 1 - e_pos

        self.draw_piece(p_pos, '#D2B48C')
        self.draw_piece(e_pos, '#5D3A3A')

        p_walls = self.state.player[1] if is_fp else self.state.enemy[1]
        e_walls = self.state.enemy[1]  if is_fp else self.state.player[1]
        self.player_walls.config(text=f"Player Walls\n{p_walls}")
        self.enemy_walls.config(text=f"Enemy Walls\n{e_walls}")

        if not is_fp:
            self.state.rotate_walls()
        self.draw_walls()
        if not is_fp:
            self.state.rotate_walls()


if __name__ == '__main__':
    root = tk.Tk()
    f = GameUI(master=root, model=model)
    f.pack()
    f.mainloop()
