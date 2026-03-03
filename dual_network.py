# ====================
# Creating the Dual Network
# ====================

# Importing packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Device selection — uses CUDA GPU if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from config import (
    USE_BFS_CHANNELS, MODEL_DIR, DATA_DIR,
    DN_FILTERS, DN_RESIDUAL_NUM,
)

DN_INPUT_SHAPE = (7, 7, 8) if USE_BFS_CHANNELS else (7, 7, 6)  # (H, W, C)
DN_OUTPUT_SIZE = 7**2 + 2*(7-1)**2  # 49 positions + 36 h-walls + 36 v-walls = 121 actions


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        sc = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + sc)


# Dual network: shared trunk -> policy head + value head
class DualNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = DN_INPUT_SHAPE[2]  # 6 input channels

        self.conv = nn.Conv2d(in_ch, DN_FILTERS, 3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        self.bn = nn.BatchNorm2d(DN_FILTERS)

        self.residuals = nn.Sequential(
            *[ResidualBlock(DN_FILTERS) for _ in range(DN_RESIDUAL_NUM)]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.policy_head = nn.Linear(DN_FILTERS, DN_OUTPUT_SIZE)
        self.value_head  = nn.Linear(DN_FILTERS, 1)

    def forward(self, x):
        # x: (N, C, H, W)
        x = F.relu(self.bn(self.conv(x)))
        x = self.residuals(x)
        x = self.pool(x).view(x.size(0), -1)  # flatten to (N, DN_FILTERS)
        p = F.softmax(self.policy_head(x), dim=1)
        v = torch.tanh(self.value_head(x))
        return p, v


# Save model weights to disk
def save_model(model, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


# Load model weights from disk and move to DEVICE
def load_model(path):
    model = DualNetwork().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


# Create and save a fresh network if one doesn't already exist
def dual_network():
    best_path = os.path.join(MODEL_DIR, 'best.pt')
    if os.path.exists(best_path):
        return

    model = DualNetwork()
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(model, best_path)
    del model


# Running the function
if __name__ == '__main__':
    dual_network()
