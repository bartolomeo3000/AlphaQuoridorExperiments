# ====================
# Central Configuration
# ====================
# All training hyperparameters live here.
# Edit this file to switch variants or tune training — nothing else needs touching.

import multiprocessing as mp

# ── Variant switch ────────────────────────────────────────────────────────────
# Set True to use 2 extra BFS goal-distance input channels.
# Requires a full restart (fresh model + data). The two variants use completely
# separate model/ and data/ folders so switching never corrupts the other run.
USE_BFS_CHANNELS = True

# ── Evaluate best player (vs hand-coded agents) ───────────────────────────────
EP_RANDOM_GAMES = 0   # Games vs random agent  (0 = skip)
EP_GREEDY_GAMES = 10   # Games vs greedy-forward agent
EP_BFS_GAMES    = 10   # Games vs BFS-shortest-path agent

# ── Evaluate network (latest vs best) ────────────────────────────────────────
EN_GAME_COUNT            = 50    # Evaluation games per cycle — must be even (paired games)
                                 # Each opening is played twice (once per side) to cancel positional bias.
EN_TEMPERATURE           = 0.0   # Temperature during MCTS play (0 = greedy; diversity comes from EN_FORCED_OPENING)
EN_TEMP_CUTOFF           = 0     # Unused when EN_TEMPERATURE=0
EN_FORCED_OPENING        = 2     # Random moves played before MCTS to guarantee diverse starting positions
EN_PROMOTE_THRESHOLD     = 0.5  # Min score for latest to replace best00
EN_DRAW_DISTANCE_SCORING = False # Score draws by BFS distance instead of flat 0.5

# ── MCTS ──────────────────────────────────────────────────────────────────────
PV_EVALUATE_COUNT  = 600    # Simulations per move during training
C_PUCT             = 1.25   # Exploration constant in PUCT formula (AlphaZero: ~1.25)
FPU_REDUCTION      = None   # First Play Urgency: unvisited Q = parent_Q - FPU_REDUCTION.
                            # None = legacy behaviour (unvisited Q = 0.0 constant).
                            # Typical range 0.1–0.4 (tanh scale) if enabled.
DIRICHLET_ALPHA    = 0.15   # Noise concentration — smaller = spread out more (Go: 0.03, Chess: 0.3)
DIRICHLET_EPSILON  = 0.30   # Fraction of prior replaced by noise (AlphaZero default; bump to 0.40 if opening entropy collapses)
POSITION_PRIOR_BOOST = 3.0  # Scale position-move priors before renormalising
                             # Counteracts wall actions dominating the action space (~30-50 walls vs ~4 position moves)
BFS_MOVE_BOOST       = 3.0   # Extra multiplier (on top of POSITION_PRIOR_BOOST) for the pawn move(s)
                             # that reduce BFS distance to the goal row. Set to 1.0 to disable.
                             # Reduced from 3.0 — BFS_PUCT_BONUS now handles in-search guidance.
BFS_MOVE_PENALTY     = 0.3   # Multiplier applied to pawn moves that INCREASE BFS distance to goal.
                             # Reduced from 0.1 — BFS_PUCT_BONUS now penalises retreats in PUCT.
BFS_ADVANCE_FLOOR    = 0.1  # After all boosts+renorm, ensure each advancing pawn move has at least
                             # this probability share. Reduced from 0.2 — PUCT bonus handles it.
BFS_RETREAT_CEILING  = 1.0  # Cap retreating pawn move priors at this value (1.0 = disabled).

BFS_PUCT_ADVANCE_BONUS  = 0.1   # PUCT bonus for pawn moves that advance toward goal.
                                # Set to 0 (default) to let the NN learn this bias itself.
BFS_PUCT_RETREAT_PENALTY= 0.2  # PUCT penalty subtracted for pawn moves that retreat.
                                # Discourages the tree from exploring backward steps.
BFS_WALL_PUCT_SCALE     = 0.01  # PUCT bonus for wall moves = scale × (Δopp - Δself).
                                # Linear: each net BFS step blocked adds a flat bonus.
BFS_PUCT_DECAY          = False  # True  → adj added to prior p, decays as sqrt(t)/(1+n).
                                # False → adj added as flat constant (old behaviour).

# ── Self-play ─────────────────────────────────────────────────────────────────
SP_GAME_COUNT       = 200    # Games generated per self-play phase  (AlphaZero paper: 25 000)
SP_TEMPERATURE      = 1.0    # Boltzmann temperature for move sampling
TEMP_CUTOFF         = 10     # Play with SP_TEMPERATURE for the first N plies, then greedy
SP_NUM_WORKERS      = max(1, mp.cpu_count() - 1)  # CPU workers; leaves one core free for OS / GPU
SP_FORCED_OPENING    = 2      # Random moves played at game start before MCTS (not recorded in history)
SP_RESIGN_THRESHOLD  = -0.97  # Resign if MCTS root Q drops below this (None = disabled)
                              # Q ≈ -0.95 → ~2.5% win probability; stricter than AlphaZero default (-0.90)
SP_RESIGN_CHECK_RATE = 0.70   # Fraction of games that skip resignation for false-resign detection.
                              # After training, false-resign candidates (Q<threshold AND game recovered)
SP_OPENING_RATE      = 0.30   # Fraction of games that start with SP_FORCED_OPENING random moves;
                              # the rest start from the initial position.
OPENING_DEPTH        = 4      # Plies tracked for opening-diversity entropy metric
SP_CKPT_EVERY  = 10     # Checkpoint self-play progress every N games (for mid-cycle resume)
POLICY_TARGET_BLEND = 0.0   # Unused / disabled.

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCH            = 100      # Epochs per training phase
BATCH_SIZE           = 256
REPLAY_BUFFER_CYCLES = 10        # How many most-recent history files to train on
LR                   = 0.001    # Adam initial learning rate
LR_MIN               = 0.00025  # Cosine-annealing floor (reached at epoch NUM_EPOCH)
WEIGHT_DECAY         = 0.0005   # L2 regularisation (equivalent to Keras kernel_regularizer)
GRAD_CLIP_NORM       = 1.0      # Max gradient norm for gradient clipping
DRAW_SHAPE_SCALE     = 0.2     # Draw value shaping weight.
                                # BFS mode:  scale*(e_bfs_dist - p_bfs_dist)/(N-1)  — wall-aware distances
                                # Row mode:  scale*(e_row - p_row)/(N-1)             — raw row proxy
                                # 0 = flat 0.5, 1 ≈ full ±1 range; used in self-play & eval data generation

# ── Network architecture ──────────────────────────────────────────────────────
DN_FILTERS      = 128   # Conv filters per layer  (AlphaZero paper: 256)
DN_RESIDUAL_NUM = 16    # Residual blocks          (AlphaZero paper: 19)
DRAW_DEPTH      = 100   # Plies before a game is declared a draw (repetition draw can trigger earlier)

# ── Paths (derived — do not edit) ─────────────────────────────────────────────
MODEL_DIR = './model_8ch' if USE_BFS_CHANNELS else './model_6ch'
DATA_DIR  = './data_8ch'  if USE_BFS_CHANNELS else './data_6ch'
LOGS_DIR  = './logs_8ch'  if USE_BFS_CHANNELS else './logs_6ch'
MODEL_SNAPSHOT_COUNT = 10  # Number of per-cycle snapshots to keep (oldest pruned automatically)
