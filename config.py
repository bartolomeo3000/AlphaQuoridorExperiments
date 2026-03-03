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
EP_GREEDY_GAMES = 5   # Games vs greedy-forward agent
EP_BFS_GAMES    = 5   # Games vs BFS-shortest-path agent

# ── Evaluate network (latest vs best) ────────────────────────────────────────
EN_GAME_COUNT            = 40    # Evaluation games per cycle  (AlphaZero paper: 400)
EN_TEMPERATURE           = 0.1   # Opening temperature during evaluation
EN_TEMP_CUTOFF           = 10    # Plies to use EN_TEMPERATURE, then greedy
EN_PROMOTE_THRESHOLD     = 0.45  # Min score for latest to replace best
EN_DRAW_DISTANCE_SCORING = False # Score draws by BFS distance instead of flat 0.5

# ── MCTS ──────────────────────────────────────────────────────────────────────
PV_EVALUATE_COUNT  = 200    # Simulations per move during training
DIRICHLET_ALPHA    = 0.15   # Noise concentration — smaller = spread out more (Go: 0.03, Chess: 0.3)
DIRICHLET_EPSILON  = 0.25   # Fraction of prior replaced by noise (AlphaZero default; bump to 0.40 if opening entropy collapses)
POSITION_PRIOR_BOOST = 4.0  # Scale position-move priors before renormalising
                             # Counteracts wall actions dominating the action space (~30-50 walls vs ~4 position moves)

# ── Self-play ─────────────────────────────────────────────────────────────────
SP_GAME_COUNT  = 200    # Games generated per self-play phase  (AlphaZero paper: 25 000)
SP_TEMPERATURE = 1.0    # Boltzmann temperature for move sampling
TEMP_CUTOFF    = 10     # Play with SP_TEMPERATURE for the first N plies, then greedy
SP_NUM_WORKERS = max(1, mp.cpu_count() - 1)  # CPU workers; leaves one core free for OS / GPU
OPENING_DEPTH  = 4      # Plies tracked for opening-diversity entropy metric
SP_CKPT_EVERY  = 10     # Checkpoint self-play progress every N games (for mid-cycle resume)

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCH            = 100      # Epochs per training phase
BATCH_SIZE           = 256
REPLAY_BUFFER_CYCLES = 7        # How many most-recent history files to train on
LR                   = 0.001    # Adam initial learning rate
LR_MIN               = 0.00025  # Cosine-annealing floor (reached at epoch NUM_EPOCH)
WEIGHT_DECAY         = 0.0005   # L2 regularisation (equivalent to Keras kernel_regularizer)
GRAD_CLIP_NORM       = 1.0      # Max gradient norm for gradient clipping
DRAW_SHAPE_SCALE     = 0.05     # Draw value = 0.5 ± DRAW_SHAPE_SCALE*(row_diff/(N-1))
                                # 0 = flat 0.5, 1 = full ±1 range; used in self-play & eval data generation

# ── Network architecture ──────────────────────────────────────────────────────
DN_FILTERS      = 128   # Conv filters per layer  (AlphaZero paper: 256)
DN_RESIDUAL_NUM = 16    # Residual blocks          (AlphaZero paper: 19)

# ── Paths (derived — do not edit) ─────────────────────────────────────────────
MODEL_DIR = './model_8ch' if USE_BFS_CHANNELS else './model_6ch'
DATA_DIR  = './data_8ch'  if USE_BFS_CHANNELS else './data_6ch'
LOGS_DIR  = './logs_8ch'  if USE_BFS_CHANNELS else './logs_6ch'
