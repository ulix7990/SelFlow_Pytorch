# data_augmentation.py

import torch
import torch.nn.functional as F
import random

__all__ = [
    'random_crop',
    'flow_vertical_flip',
    'flow_horizontal_flip',
    'random_flip',
    'random_flip_with_flow',
    'random_channel_swap',
    'flow_resize',
]

def random_crop(tensor_list, crop_h: int, crop_w: int):
    """
    Apply the same random crop to a list of tensors.
    Each tensor is assumed to have shape (C, H, W).
    """
    H, W = tensor_list[0].shape[-2:]
    top = torch.randint(0, H - crop_h + 1, (1,)).item()
    left = torch.randint(0, W - crop_w + 1, (1,)).item()
    return [
        t[..., top : top + crop_h, left : left + crop_w]
        for t in tensor_list
    ]


def flow_vertical_flip(flow: torch.Tensor) -> torch.Tensor:
    """
    Flip the flow vertically and invert the vertical component.
    flow: Tensor of shape (2, H, W)
    """
    # flip along height
    f = torch.flip(flow, dims=[-2])
    u, v = f[0], f[1] * -1
    return torch.stack([u, v], dim=0)


def flow_horizontal_flip(flow: torch.Tensor) -> torch.Tensor:
    """
    Flip the flow horizontally and invert the horizontal component.
    flow: Tensor of shape (2, H, W)
    """
    # flip along width
    f = torch.flip(flow, dims=[-1])
    u, v = f[0] * -1, f[1]
    return torch.stack([u, v], dim=0)


def random_flip(tensor_list):
    """
    Randomly flip each tensor in the list horizontally and/or vertically.
    tensor_list: list of Tensors, each (C, H, W)
    """
    do_h = bool(torch.randint(0, 2, (1,)).item())
    do_v = bool(torch.randint(0, 2, (1,)).item())
    out = []
    for t in tensor_list:
        if do_h:
            t = torch.flip(t, dims=[-1])  # horizontal
        if do_v:
            t = torch.flip(t, dims=[-2])  # vertical
        out.append(t)
    return out


def random_flip_with_flow(img_list, flow_list):
    """
    Randomly flip image tensors and corresponding flow tensors.
    img_list: list of image Tensors (C, H, W)
    flow_list: list of flow Tensors (2, H, W)
    Returns (flipped_imgs, flipped_flows)
    """
    do_h = bool(torch.randint(0, 2, (1,)).item())
    do_v = bool(torch.randint(0, 2, (1,)).item())

    imgs = []
    for img in img_list:
        if do_h:
            img = torch.flip(img, dims=[-1])
        if do_v:
            img = torch.flip(img, dims=[-2])
        imgs.append(img)

    flows = []
    for flow in flow_list:
        if do_h:
            flow = flow_horizontal_flip(flow)
        if do_v:
            flow = flow_vertical_flip(flow)
        flows.append(flow)

    return imgs, flows


def random_channel_swap(img_list):
    """
    Randomly permute the 3 color channels of each image in the list.
    img_list: list of image Tensors, each (3, H, W)
    """
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    perm = random.choice(perms)
    return [img[list(perm), ...] for img in img_list]


def flow_resize(flow: torch.Tensor,
                out_size: tuple,
                is_scale: bool = True,
                mode: str = 'bilinear',
                align_corners: bool = True) -> torch.Tensor:
    """
    Resize a flow field to out_size = (H_out, W_out).
    flow: Tensor of shape (2, H, W)
    """
    H0, W0 = flow.shape[-2:]
    f = flow.unsqueeze(0)  # (1, 2, H, W)
    f_resized = F.interpolate(f, size=out_size, mode=mode, align_corners=align_corners)
    f_resized = f_resized.squeeze(0)

    if is_scale:
        scale_h = out_size[0] / H0
        scale_w = out_size[1] / W0
        # channel 0: horizontal displacement, channel 1: vertical
        f_resized[0] *= scale_w
        f_resized[1] *= scale_h

    return f_resized
