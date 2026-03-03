# ====================
# parameter update part
# ====================

import os
from dual_network import DN_INPUT_SHAPE, DualNetwork, load_model, save_model, DEVICE
from config import (
    NUM_EPOCH, BATCH_SIZE, REPLAY_BUFFER_CYCLES,
    LR, LR_MIN, WEIGHT_DECAY, GRAD_CLIP_NORM,
    MODEL_DIR, DATA_DIR,
)
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler

# Policy index permutation for a left-right board flip — computed dynamically so it
# works for any board size. Verified to be a self-inverse permutation (applying it
# twice returns the identity).
def _compute_flip_lr_policy():
    N = DN_INPUT_SHAPE[0]  # board height/width
    perm = []
    # Positions 0..N²-1: (row, col) -> (row, N-1-col)
    for r in range(N):
        for c in range(N):
            perm.append(r * N + (N - 1 - c))
    # H-walls N²..N²+(N-1)²-1: (row, col) -> (row, N-2-col)
    offset_h = N * N
    for r in range(N - 1):
        for c in range(N - 1):
            perm.append(offset_h + r * (N - 1) + (N - 2 - c))
    # V-walls N²+(N-1)²..N²+2*(N-1)²-1: same flip
    offset_v = N * N + (N - 1) * (N - 1)
    for r in range(N - 1):
        for c in range(N - 1):
            perm.append(offset_v + r * (N - 1) + (N - 2 - c))
    return np.array(perm)

_FLIP_LR_POLICY = _compute_flip_lr_policy()


def augment(s, p, v):
    """Double the dataset with a left-right board flip."""
    s_flip = np.flip(s, axis=3).copy()       # (N, C, H, W) -> flip W dimension
    p_flip = p[:, _FLIP_LR_POLICY]           # permute policy actions accordingly
    return (np.concatenate([s, s_flip]),
            np.concatenate([p, p_flip]),
            np.concatenate([v, v]))


def load_data():
    history_paths = sorted(Path(DATA_DIR).glob('*.history'))[-REPLAY_BUFFER_CYCLES:]
    history = []
    for path in history_paths:
        with path.open(mode='rb') as f:
            history.extend(pickle.load(f))
    return history


# Training the dual network
def train_network():
    # Loading training data
    history = load_data()
    s, p, v = zip(*history)

    # Reshape to (N, C, H, W) for PyTorch — no NHWC transpose needed
    a, b, c = DN_INPUT_SHAPE  # a=H=3, b=W=3, c=C=6
    s = np.array(s, dtype=np.float32).reshape(len(s), c, a, b)
    p = np.array(p, dtype=np.float32)
    v = np.array(v, dtype=np.float32).reshape(-1, 1)

    # Data augmentation: left-right flip doubles the training set for free
    s, p, v = augment(s, p, v)
    print(f'Training on {len(s)} positions ({len(s)//2} original + {len(s)//2} augmented)')

    # Move data to tensors
    s = torch.from_numpy(s)
    p = torch.from_numpy(p)
    v = torch.from_numpy(v)

    dataset = TensorDataset(s, p, v)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Loading the best player's model
    model = load_model(os.path.join(MODEL_DIR, 'best.pt'))
    model.train()

    # Optimiser — weight_decay provides L2 regularisation equivalent to Keras' kernel_regularizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Mixed precision scaler — ~1.5-2x speedup on modern NVIDIA GPUs
    use_amp = (DEVICE.type == 'cuda')
    scaler  = GradScaler('cuda') if use_amp else None

    # Cosine annealing LR — smooth decay from 0.001 to 0.00025 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=LR_MIN)

    first_avg_l = None  # for loss sanity check (start vs end)

    for epoch in range(NUM_EPOCH):
        epoch_loss_p = 0.0
        epoch_loss_v = 0.0
        num_batches = 0
        for s_batch, p_batch, v_batch in loader:
            s_batch = s_batch.to(DEVICE)
            p_batch = p_batch.to(DEVICE)
            v_batch = v_batch.to(DEVICE)

            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                p_pred, v_pred = model(s_batch)
                # Soft cross-entropy for policy, MSE for value
                loss_p = -torch.mean(torch.sum(p_batch * torch.log(p_pred + 1e-8), dim=1))
                loss_v = nn.functional.mse_loss(v_pred, v_batch)
                loss   = loss_p + loss_v

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                optimizer.step()

            epoch_loss_p += loss_p.item()
            epoch_loss_v += loss_v.item()
            num_batches += 1

        avg_lp = epoch_loss_p / num_batches
        avg_lv = epoch_loss_v / num_batches
        avg_l  = avg_lp + avg_lv

        if first_avg_l is None:
            first_avg_l = avg_l

        scheduler.step()
        print(f'\rTraining epoch {epoch+1:>3}/{NUM_EPOCH}  '
              f'loss={avg_l:.4f}  loss_policy={avg_lp:.4f}  loss_value={avg_lv:.4f}', end='')
    print(f'\nTraining done — final loss={avg_l:.4f}  loss_policy={avg_lp:.4f}  loss_value={avg_lv:.4f}')
    if avg_l > first_avg_l:
        print(f'WARNING: final loss ({avg_l:.4f}) > initial loss ({first_avg_l:.4f}) — '
              f'training failed to converge (bad LR, corrupted data, or gradient explosion)')

    # Saving the latest player's model
    save_model(model, os.path.join(MODEL_DIR, 'latest.pt'))

    # Releasing the model
    del model

    return {'loss': round(avg_l, 4), 'loss_policy': round(avg_lp, 4), 'loss_value': round(avg_lv, 4)}


if __name__ == '__main__':
    train_network()
