# ====================
# Execution of Learning Cycle
# ====================

# Importing packages
from dual_network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
from evaluate_best_player import evaluate_best_player
from datetime import datetime
import csv
import glob
import shutil
import sys
import os
import time

STOP_FLAG_FILENAME = '.stop_requested'

def _stop_flag_path():
    from config import LOGS_DIR
    return os.path.join(LOGS_DIR, STOP_FLAG_FILENAME)


class _Tee:
    """Write to both the original stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, 'a', encoding='utf-8', buffering=1)

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

# Number of training cycles
NUM_TRAIN_CYCLE = 30


_CSV_HEADERS = [
    'cycle', 'timestamp',
    'sp_W_pct', 'sp_D_pct', 'sp_L_pct', 'sp_positions', 'sp_unique', 'sp_entropy', 'sp_top1',
    'sp_avg_game_len', 'sp_avg_walls', 'sp_resign_pct', 'sp_false_resign',
    'loss', 'loss_policy', 'loss_value',
    'eval_score', 'promoted',
    'en_entropy', 'en_unique', 'en_top1',
    'vs_random', 'vs_greedy', 'vs_bfs',
    't_selfplay_min', 't_train_min', 't_evalnet_min', 't_evalbest_min', 't_cycle_min',
]

def _append_stats(stats_path, row: dict):
    write_header = not os.path.exists(stats_path)
    with open(stats_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _run(stats_path):
    # Creating the dual network
    dual_network()

    # Resume cycle numbering from where a previous run left off
    cycle_offset = 0
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if rows:
            cycle_offset = int(rows[-1]['cycle']) + 1
    if cycle_offset:
        print(f'Resuming from cycle {cycle_offset}')

    training_start = time.time()

    for i in range(NUM_TRAIN_CYCLE):
        cycle_num = cycle_offset + i
        cycle_start = time.time()
        ts = datetime.now().strftime('%H:%M:%S')
        print(f'\n==================== Cycle {cycle_num:>3}  [{ts}] ====================')

        # self-play part
        t0 = time.time()
        sp_stats = self_play(cycle_num=cycle_num)
        t_sp = (time.time() - t0) / 60
        print(f'[timing] self-play:    {t_sp:.1f} min')

        # parameter update part
        t0 = time.time()
        tr_stats = train_network()
        t_tr = (time.time() - t0) / 60
        print(f'[timing] training:     {t_tr:.1f} min')

        # Save a numbered snapshot of latest.pt and prune old ones
        from config import MODEL_DIR, MODEL_SNAPSHOT_COUNT
        snap_dst = os.path.join(MODEL_DIR, f'cycle_{cycle_num:04d}.pt')
        shutil.copy2(os.path.join(MODEL_DIR, 'latest.pt'), snap_dst)
        snapshots = sorted(glob.glob(os.path.join(MODEL_DIR, 'cycle_*.pt')))
        for old in snapshots[:-MODEL_SNAPSHOT_COUNT]:
            os.remove(old)
            print(f'[snapshots] pruned {os.path.basename(old)}')

        # Evaluating new parameters
        t0 = time.time()
        promoted, ev_stats = evaluate_network(cycle_num=cycle_num)
        t_en = (time.time() - t0) / 60
        print(f'[timing] eval-network: {t_en:.1f} min')

        # Evaluating the best player
        bp_stats = {}
        t_bp = 0.0
        if promoted:
            t0 = time.time()
            bp_stats = evaluate_best_player()
            t_bp = (time.time() - t0) / 60
            print(f'[timing] eval-best:    {t_bp:.1f} min')

        cycle_elapsed = time.time() - cycle_start
        total_elapsed = time.time() - training_start
        total_h, total_rem = divmod(int(total_elapsed), 3600)
        total_m, total_s   = divmod(total_rem, 60)
        print(f'--- Cycle {cycle_num} done in {cycle_elapsed/60:.1f} min  |  Total: {total_h:02d}:{total_m:02d}:{total_s:02d} ---')

        # Write one CSV row summarising this cycle
        _append_stats(stats_path, {
            'cycle':        cycle_num,
            'timestamp':    ts,
            'sp_W_pct':     round(sp_stats.get('W_pct', 0), 1),
            'sp_D_pct':     round(sp_stats.get('D_pct', 0), 1),
            'sp_L_pct':     round(sp_stats.get('L_pct', 0), 1),
            'sp_positions':    sp_stats.get('positions', ''),
            'sp_unique':        sp_stats.get('unique', ''),
            'sp_entropy':       sp_stats.get('entropy', ''),
            'sp_top1':          sp_stats.get('top1_count', ''),
            'sp_avg_game_len':    sp_stats.get('avg_game_len', ''),
            'sp_avg_walls':       sp_stats.get('avg_walls', ''),
            'sp_resign_pct':      sp_stats.get('resign_pct', ''),
            'sp_false_resign':    sp_stats.get('false_resign_count', ''),
            'loss':         tr_stats.get('loss', ''),
            'loss_policy':  tr_stats.get('loss_policy', ''),
            'loss_value':   tr_stats.get('loss_value', ''),
            'eval_score':   ev_stats.get('score', ''),
            'promoted':     int(promoted),
            'en_entropy':   ev_stats.get('entropy', ''),
            'en_unique':    ev_stats.get('unique', ''),
            'en_top1':      ev_stats.get('top1_count', ''),
            'vs_random':    bp_stats.get('vs_random') if bp_stats.get('vs_random') is not None else '',
            'vs_greedy':    bp_stats.get('vs_greedy') if bp_stats.get('vs_greedy') is not None else '',
            'vs_bfs':       bp_stats.get('vs_bfs') if bp_stats.get('vs_bfs') is not None else '',
            't_selfplay_min':  round(t_sp, 1),
            't_train_min':     round(t_tr, 1),
            't_evalnet_min':   round(t_en, 1),
            't_evalbest_min':  round(t_bp, 1),
            't_cycle_min':     round(cycle_elapsed / 60, 1),
        })

        # Check for graceful-stop flag written by the web dashboard
        stop_flag = _stop_flag_path()
        if os.path.exists(stop_flag):
            try:
                os.remove(stop_flag)
            except OSError:
                pass
            print('[training] Stop flag detected — exiting after this cycle.')
            break


# Main function
if __name__ == '__main__':
    from config import LOGS_DIR
    # Redirect all prints to both console and a timestamped log file
    _log_path = os.path.join(LOGS_DIR, datetime.now().strftime('%Y%m%d_%H%M%S') + '.log')
    _stats_path = os.path.join(LOGS_DIR, 'stats.csv')
    os.makedirs(LOGS_DIR, exist_ok=True)
    _tee = _Tee(_log_path)
    sys.stdout = _tee
    print(f'Logging to {_log_path}')
    from config import USE_BFS_CHANNELS, MODEL_DIR, DATA_DIR, LOGS_DIR
    from dual_network import DN_INPUT_SHAPE
    ch = DN_INPUT_SHAPE[2]
    bfs_tag = 'BFS channels ON' if USE_BFS_CHANNELS else 'BFS channels OFF'
    print(f'Variant: {ch}-channel ({bfs_tag})  |  model={MODEL_DIR}  data={DATA_DIR}')

    try:
        _run(_stats_path)
    finally:
        _tee.close()
