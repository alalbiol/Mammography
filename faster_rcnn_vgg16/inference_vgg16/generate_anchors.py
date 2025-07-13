
import torch


# --- Helper functions for the corrected generate_anchors_pytorch ---
def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    # Ensure ws and hs are 2D for broadcasting with x_ctr, y_ctr
    ws = ws.unsqueeze(1) # Add a new dimension
    hs = hs.unsqueeze(1) # Add a new dimension
    anchors = torch.cat((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)), dim=1)
    return anchors

def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    # Rounding is important here for exact match
    ws = torch.round(torch.sqrt(size_ratios))
    hs = torch.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                                    scales=torch.tensor([8, 16, 32], dtype=torch.float32)):
    base_anchor = torch.tensor([1, 1, base_size, base_size], dtype=torch.float32) - 1 # [0, 0, 15, 15]
    
    # First, enumerate ratio anchors
    ratio_anchors = _ratio_enum(base_anchor, torch.tensor(ratios, dtype=torch.float32))
    
    all_anchors = []
    for i in range(ratio_anchors.shape[0]):
        # Then, for each ratio anchor, enumerate scales
        scaled_anchors = _scale_enum(ratio_anchors[i, :], scales)
        all_anchors.append(scaled_anchors)
    
    return torch.cat(all_anchors, dim=0)
