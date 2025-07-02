# warp.py

import torch
import torch.nn.functional as F

def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp image or feature map according to optical flow.
    
    Args:
        img: Tensor of shape (B, C, H, W)
        flow: Tensor of shape (B, 2, H, W), flow in pixel units (dx, dy)
    
    Returns:
        out: Tensor of shape (B, C, H, W) — img warped by flow
    """
    B, C, H, W = img.shape
    # 1) Create base grid of pixel coordinates (x, y)
    device, dtype = img.device, flow.dtype
    # meshgrid gives y first, then x, so swap
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=device, dtype=dtype),
        torch.arange(0, W, device=device, dtype=dtype),
        indexing='ij'
    )  # both (H, W)
    base_grid = torch.stack((grid_x, grid_y), dim=2)          # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)   # (B, H, W, 2)

    # 2) Add flow (permute to (B, H, W, 2))
    flow_grid = flow.permute(0, 2, 3, 1)                       # (B, H, W, 2)
    vgrid = base_grid + flow_grid                              # (B, H, W, 2)

    # 3) Normalize to [-1, 1] for grid_sample
    vgrid_x = 2.0 * vgrid[..., 0] / (W - 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / (H - 1) - 1.0
    vgrid_norm = torch.stack((vgrid_x, vgrid_y), dim=3)        # (B, H, W, 2)

    # 4) Sample
    # padding_mode='border' 로 경계 밖 값을 가장자리 픽셀로 처리
    out = F.grid_sample(img, vgrid_norm, mode='bilinear',
                        padding_mode='border', align_corners=True)
    return out
