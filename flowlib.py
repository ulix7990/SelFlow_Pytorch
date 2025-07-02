# flowlib.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import re
import sys

def read_flo(filename: str) -> np.ndarray:
    """
    Read Middlebury .flo format.
    Returns numpy array of shape (H, W, 2).
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError(f'Invalid .flo file: magic={magic}')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    return np.reshape(data, (h, w, 2))


def write_flo(filename: str, flow: np.ndarray) -> None:
    """
    Write numpy flow (H, W, 2) to Middlebury .flo format.
    """
    with open(filename, 'wb') as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([flow.shape[1]], dtype=np.int32).tofile(f)
        np.array([flow.shape[0]], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def read_pfm(path: str) -> np.ndarray:
    """
    Read PFM file. Returns (H, W, 2) float32 array.
    """
    with open(path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        color = (header == 'PF')
        dim_line = f.readline().decode('utf-8')
        m = re.match(r'^(\d+)\s+(\d+)\s*$', dim_line)
        if not m:
            raise ValueError('Malformed PFM header.')
        width, height = map(int, m.groups())
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        data = np.fromfile(f, endian+'f')
    shape = (height, width, 3) if color else (height, width)
    img = np.reshape(data, shape)
    img = np.flipud(img)
    return img[..., :2].astype(np.float32)


def write_pfm(path: str, image: np.ndarray, scale: float = 1.0) -> None:
    """
    Write image (H, W, 2 or 3) as PFM. If 2-channel, writes as PF with 2 channels.
    """
    if image.dtype.name != 'float32':
        raise ValueError('Image dtype must be float32.')
    img = np.flipud(image)
    color = (img.ndim == 3 and img.shape[2] >= 3)
    with open(path, 'wb') as f:
        f.write(('PF\n' if color else 'Pf\n').encode('utf-8'))
        f.write(f'{img.shape[1]} {img.shape[0]}\n'.encode('utf-8'))
        endian_char = '<' if sys.byteorder=='little' else '>'
        scl = -scale if endian_char=='<' else scale
        f.write(f'{scl}\n'.encode('utf-8'))
        img.tofile(f)


def flow_to_color(flow: np.ndarray,
                  mask: np.ndarray = None,
                  max_flow: float = None) -> np.ndarray:
    """
    Convert flow to RGB image via HSV mapping.
    - flow: (H, W, 2) or (B, H, W, 2)
    - mask: same spatial dims, valid=1
    Returns float image in [0,1], same spatial shape + 3-ch.
    """
    # handle batch dimension
    batched = (flow.ndim == 4)
    if not batched:
        flow = flow[None, ...]
    B, H, W, _ = flow.shape

    if mask is None:
        mask = np.ones((B, H, W), dtype=bool)
    else:
        mask = mask.astype(bool).reshape(B, H, W)

    # compute magnitude & angle
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u*u + v*v)
    ang = np.arctan2(v, u)  # [-π, π]

    if max_flow is None:
        max_flow = np.max(mag[mask])
    max_flow = max(max_flow, 1e-9)

    # HSV components
    h = (ang / (2*np.pi) + 1.0) % 1.0
    s = np.clip(mag * 8.0 / max_flow, 0, 1)
    v_ = np.clip(1.0 - s, 0, 1)
    hsv = np.stack([h, s, v_], axis=-1)  # (B,H,W,3)

    # convert and apply mask
    rgb = mcolors.hsv_to_rgb(hsv.reshape(-1,3)).reshape(B, H, W, 3)
    rgb[~mask[...,None]] *= 0.0
    return rgb[0] if not batched else rgb


def flow_error_image(flow1: np.ndarray,
                     flow2: np.ndarray,
                     mask_occ: np.ndarray,
                     mask_noc: np.ndarray = None,
                     log_colors: bool = True) -> np.ndarray:
    """
    Visualize flow error:
    - flow1, flow2: (H, W, 2) or (B, H, W, 2)
    - mask_occ: valid mask (1=valid)
    - mask_noc: non-occluded valid mask
    Returns RGB error image in [0,1].
    """
    if flow1.ndim == 3:
        flow1 = flow1[None]
        flow2 = flow2[None]
        mask_occ = mask_occ[None]
        if mask_noc is not None:
            mask_noc = mask_noc[None]
    B, H, W, _ = flow1.shape

    if mask_noc is None:
        mask_noc = mask_occ.copy()

    diff = np.linalg.norm(flow1 - flow2, axis=-1, keepdims=True)  # (B,H,W,1)

    if log_colors:
        # KITTI colormap bins
        cmap = np.array([
            [0,0.0625,49,54,149],
            [0.0625,0.125,69,117,180],
            [0.125,0.25,116,173,209],
            [0.25,0.5,171,217,233],
            [0.5,1,224,243,248],
            [1,2,254,224,144],
            [2,4,253,174,97],
            [4,8,244,109,67],
            [8,16,215,48,39],
            [16,1e9,165,0,38],
        ], dtype=np.float32)
        cmap[:,2:5] /= 255.0

        mag2 = np.linalg.norm(flow2, axis=-1, keepdims=True)
        err = np.minimum(diff/3.0, 20.0*diff/(mag2+1e-9))

        im = np.zeros((B, H, W, 3), dtype=np.float32)
        for low, high, r, g, b in cmap:
            mask_bin = (err>=low) & (err<high)
            im[mask_bin[...,0]] = [r, g, b]

        # dim non-noc pixels, zero-out invalid
        im = np.where(mask_noc[...,None], im, im*0.5)
        im *= mask_occ[...,None]
    else:
        # simple error map
        err_norm = np.minimum(diff,5.0)/5.0
        err_norm *= mask_occ[...,None]
        r = err_norm
        g = err_norm * mask_noc[...,None]
        b = err_norm * mask_noc[...,None]
        im = np.concatenate([r, g, b], axis=-1)

    return im[0] if flow1.shape[0]==1 else im
