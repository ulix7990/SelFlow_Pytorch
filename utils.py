# utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from warp import warp  # warp(img, flow) 함수를 임포트

__all__ = [
    'mvn', 'lrelu', 'imshow', 'rgb_bgr',
    'compute_Fl', 'length_sq', 'occlusion'
]

def mvn(img: torch.Tensor) -> torch.Tensor:
    """
    Mean–variance normalization.
    img: Tensor of shape (..., C, H, W) 또는 (C, H, W).
    채널별로 spatial dimension의 평균/분산을 계산해 정규화.
    """
    # spatial dims 마지막 두 개
    spatial_dims = tuple(range(img.dim() - 2, img.dim()))
    mean = img.mean(dim=spatial_dims, keepdim=True)
    var = img.var(dim=spatial_dims, unbiased=False, keepdim=True)
    return (img - mean) / torch.sqrt(var + 1e-12)


def lrelu(x: torch.Tensor, leak: float = 0.2) -> torch.Tensor:
    """
    LeakyReLU activation.
    """
    return torch.maximum(x, leak * x)


def imshow(img, re_normalize: bool = False):
    """
    numpy array 또는 torch.Tensor 이미지를 화면에 출력.
    - img: H×W or H×W×C, torch.Tensor일 경우 CPU로 옮긴 뒤 numpy로 변환.
    - re_normalize: True면 [min,max] → [0,255] 스케일링.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if re_normalize:
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn + 1e-6) * 255

    # 흑백인 경우 3채널로 복제
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    plt.show()


def rgb_bgr(img: np.ndarray) -> np.ndarray:
    """
    OpenCV 호환 BGR ↔ RGB 변환.
    """
    out = img.copy()
    out[..., [0, 2]] = out[..., [2, 0]]
    return out


def compute_Fl(flow_gt: torch.Tensor,
               flow_est: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    TensorFlow 의 compute_Fl → PyTorch port.
    err_norm > 3 AND err_norm/|flow_gt| > 0.05 인 픽셀의 비율.
    - flow_*: (B, 2, H, W)
    - mask:    (B, 1, H, W) 또는 (B, H, W)
    """
    # 브로드캐스트 마스크 shape 맞추기
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    err = (flow_gt - flow_est) * mask
    # 픽셀별 벡터 길이
    err_norm    = torch.norm(err, p=2, dim=1)  # (B, H, W)
    gt_norm     = torch.norm(flow_gt, p=2, dim=1).clamp(min=1e-12)
    # 기준 조건
    cond1 = err_norm > 3
    cond2 = (err_norm / gt_norm) > 0.05
    logic = cond1 & cond2
    logic = logic.unsqueeze(1) & (mask > 0)
    # F1 = (조건 만족한 유효 픽셀 수) / (마스크 픽셀 수)
    return logic.sum(dtype=torch.float32) / (mask.sum() + 1e-6)


def length_sq(x: torch.Tensor) -> torch.Tensor:
    """
    채널(특성) dimension 제곱합.
    - x: (B, C, H, W) → returns (B, 1, H, W)
    """
    return torch.sum(x * x, dim=1, keepdim=True)


def occlusion(flow_fw: torch.Tensor,
              flow_bw: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    양방향 광학 흐름의 occlusion mask 계산.
    - flow_fw, flow_bw: (B, 2, H, W)
    Returns:
      occ_fw, occ_bw: 두 텐서 모두 (B, 1, H, W), occluded 영역=1
    """
    B, C, H, W = flow_fw.shape
    # 반대 방향 흐름을 warp
    flow_bw_warped = warp(flow_bw, flow_fw)  # (B,2,H,W)
    flow_fw_warped = warp(flow_fw, flow_bw)

    diff_fw = flow_fw + flow_bw_warped
    diff_bw = flow_bw + flow_fw_warped

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    thresh_fw = 0.01 * mag_sq_fw + 0.5
    thresh_bw = 0.01 * mag_sq_bw + 0.5

    occ_fw = (length_sq(diff_fw) > thresh_fw).float()
    occ_bw = (length_sq(diff_bw) > thresh_bw).float()
    return occ_fw, occ_bw
