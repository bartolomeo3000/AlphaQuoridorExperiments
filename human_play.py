# Importing necessary packages and modules
import os
from game import State
from pv_mcts import pv_mcts_action, random_action
from dual_network import load_model
from config import MODEL_DIR
import tkinter as tk
import pv_mcts
pv_mcts.PV_EVALUATE_COUNT = 2000  # stronger play vs human; training stays at 200

# Loading the best player's model
model = load_model(os.path.join(MODEL_DIR, 'best.pt'))

# Defining the Game UI
class GameUI(tk.Frame):
    # Initialization
    def __init__(self, master=None, model=None):
        tk.Frame.__init__(self, master)
        self.master.title('Quoridor')

        # Generating the game state
        self.state = State()
        self.N = self.state.N
        self.D = 80  # default cell size; updated dynamically on resize

        self.select = -1  # Selection (-1: none, 0~(N*N-1): square)
        self.placing_wall = False  # Flag to indicate if we are placing a wall

        # Creating the function for action selection using PV MCTS
        self.next_action = pv_mcts_action(model) if model else random_action()

        # Make the frame fill the window
        self.grid(sticky='nsew')
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Creating the canvas for the game board — no fixed size, expands with window
        self.c = tk.Canvas(self, highlightthickness=0, bg='#4B4B4B')
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.bind('<Configure>', self._on_resize)
        self.c.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        # Displaying the player's walls on the left
        self.player_walls_frame = tk.Frame(self)
        self.player_walls_frame.grid(row=1, column=2, padx=10, pady=10)
        self.player_walls = tk.Label(self.player_walls_frame, text="Player Walls", anchor="center", justify=tk.CENTER, font=('Helvetica', 14))
        self.player_walls.pack()

        # Displaying the enemy's walls on the right
        self.enemy_walls_frame = tk.Frame(self)
        self.enemy_walls_frame.grid(row=1, column=0, padx=10, pady=10)
        self.enemy_walls = tk.Label(self.enemy_walls_frame, text="Enemy Walls", anchor="center", justify=tk.CENTER, font=('Helvetica', 14))
        self.enemy_walls.pack()

        # Displaying the action buttons below the game board
        self.controls_frame = tk.Frame(self)
        self.controls_frame.grid(row=2, column=1, padx=10, pady=10)
        self.wall_button = tk.Button(self.controls_frame, text="Place Wall", command=self.place_wall_mode)
        self.wall_button.pack()

        self.wall_direction = tk.StringVar(value="horizontal")
        self.wall_horizontal_button = tk.Radiobutton(self.controls_frame, text="Horizontal", variable=self.wall_direction, value="horizontal")
        self.wall_vertical_button = tk.Radiobutton(self.controls_frame, text="Vertical", variable=self.wall_direction, value="vertical")
        self.wall_horizontal_button.pack()
        self.wall_vertical_button.pack()

        # Result message
        self.result_message = tk.Label(self, text="", font=('Helvetica', 24))
        self.result_message.grid(row=0, column=1, pady=10)

        # Set a sensible initial window size (60% of screen, square-ish)
        screen_w = master.winfo_screenwidth()
        screen_h = master.winfo_screenheight()
        init = min(int(screen_w * 0.6), int(screen_h * 0.8))
        master.geometry(f'{init}x{int(init * 0.85)}')

        # Updating the drawing
        self.on_draw()

    def _on_resize(self, event):
        """Recompute cell size from actual canvas dimensions and redraw."""
        board_px = min(event.width, event.height)
        self.D = max(20, board_px // self.N)
        self.on_draw()

    def place_wall_mode(self):
        self.placing_wall = not self.placing_wall
        self.wall_button.config(text="Move Piece" if self.placing_wall else "Place Wall")

    # Human's turn
    def turn_of_human(self, event):
        N = self.N
        D = self.D
        # If the game is over
        if self.state.is_done():
            return

        # If it is not the first player's turn
        if not self.state.is_first_player():
            return

        # Calculate the selection and move position
        if self.placing_wall:
            x, y = (event.x - D // 2) // D, (event.y - D // 2) // D
            print(x, y)
            if 0 <= x < N - 1 and 0 <= y < N - 1:
                if not self.place_wall(x, y):
                    return  # illegal wall — don't give AI a free turn
            else:
                return  # click out of bounds — ignore
        else:
            x, y = event.x // D, event.y // D
            self.select = N * y + x
            action = self.select

            # Convert selection and move to action

            # If the action is not legal
            if not (action in self.state.legal_actions()):
                self.select = -1
                self.on_draw()
                return

            # Get the next state
            self.state = self.state.next(action)
            self.select = -1
            self.on_draw()

        # AI's turn
        self.master.after(500, self.turn_of_ai)

    def place_wall(self, x, y):
        N = self.N
        # Adjusted logic for placing walls at grid points
        if self.wall_direction.get() == "horizontal":
            action = N ** 2 + (N - 1) * y + x
        else:
            action = N ** 2 + (N - 1) ** 2 + (N - 1) * y + x

        # Check if the action is legal
        if action in self.state.legal_actions():
            # Get the next state
            self.state = self.state.next(action)
            self.placing_wall = False
            self.wall_button.config(text="Place Wall")
            self.on_draw()
            return True  # legal — AI should take its turn
        else:
            self.on_draw()
            return False  # illegal — human retries, AI does not move

    # AI's turn
    def turn_of_ai(self):
        # If the game is over
        if self.state.is_done():
            self.display_result()
            self.master.after(1000, self.reset_game)
            return

        # Get the action
        action = self.next_action(self.state)

        # Get the next state
        self.state = self.state.next(action)
        self.on_draw()

        if self.state.is_done():
            self.display_result()
            self.master.after(1000, self.reset_game)
            return

    def display_result(self):
        if self.state.is_draw():
            self.result_message.config(text="Draw", fg="gray")
            return
        is_lose = self.state.is_lose() if self.state.is_first_player() else not self.state.is_lose()
        if is_lose:
            self.result_message.config(text="You Lose", fg="blue")
        else:
            self.result_message.config(text="You Win", fg="red")
    
    def reset_game(self):
        self.state = State()
        self.on_draw()
        self.result_message.config(text="")

    # Draw the piece
    def draw_piece(self, index, color):
        N = self.N
        D = self.D
        x = (index % N) * D
        y = (index // N) * D
        margin = D // 10
        self.c.create_oval(x + margin, y + margin, x + D - margin, y + D - margin, fill=color, outline='black')

    # Draw the walls
    def draw_walls(self):
        N = self.N
        D = self.D
        for i in range(len(self.state.walls)):
            x, y = i % (N - 1), i // (N - 1)
            if self.state.walls[i] == 1:
                x1, y1 = x * D, (y + 1) * D
                x2, y2 = (x + 2) * D, (y + 1) * D
                self.c.create_line(x1, y1, x2, y2, width=max(4, D // 12), fill='#D1B575')
            elif self.state.walls[i] == 2:
                x1, y1 = (x + 1) * D, y * D
                x2, y2 = (x + 1) * D, (y + 2) * D
                self.c.create_line(x1, y1, x2, y2, width=max(4, D // 12), fill='#D1B575')

    # Update the drawing
    def on_draw(self):
        N = self.N
        D = self.D
        L = N * D
        is_first_player = self.state.is_first_player()

        # Grid
        self.c.delete('all')
        self.c.create_rectangle(0, 0, L, L, width=0.0, fill='#4B4B4B')
        for i in range(1, N):
            self.c.create_line(i * D, 0, i * D, L, width=max(4, D // 12), fill='#8B0000')
            self.c.create_line(0, i * D, L, i * D, width=max(4, D // 12), fill='#8B0000')

        # Pieces
        p_pos = self.state.player[0] if is_first_player else self.state.enemy[0]
        e_pos = self.state.enemy[0] if is_first_player else self.state.player[0]
        e_pos = N ** 2 - 1 - e_pos

        self.draw_piece(p_pos, '#D2B48C')
        self.draw_piece(e_pos, '#5D3A3A')

        p_walls = self.state.player[1] if is_first_player else self.state.enemy[1]
        e_walls = self.state.enemy[1] if is_first_player else self.state.player[1]

        # Update the wall count
        self.player_walls.config(text=f"Player Walls\n{p_walls}")
        self.enemy_walls.config(text=f"Enemy Walls\n{e_walls}")

        if not is_first_player:
            self.state.rotate_walls()

        # Walls
        self.draw_walls()

        if not is_first_player:
            self.state.rotate_walls()

# Run the game UI
if __name__ == '__main__':
    root = tk.Tk()
    f = GameUI(master=root, model=model)
    f.pack()
    f.mainloop()
