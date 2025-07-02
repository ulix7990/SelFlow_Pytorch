# network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_augmentation import flow_resize
from utils import lrelu
from warp import warp

def cost_volume(x1: torch.Tensor,
                x2: torch.Tensor,
                max_disp: int = 4) -> torch.Tensor:
    """
    Compute correlation-based cost volume between x1 and x2.
    Args:
        x1, x2: feature maps of shape (B, C, H, W)
        max_disp: maximum displacement (pixels) to search in each direction
    Returns:
        cost: (B, (2*max_disp+1)**2, H, W)
    """
    B, C, H, W = x1.shape
    d = 2 * max_disp + 1

    # pad x2 so we can extract local patches around each pixel
    pad = max_disp
    x2_padded = F.pad(x2, (pad, pad, pad, pad))  # (B, C, H+2p, W+2p)

    # extract (d x d) neighborhood for each pixel
    # shape → (B, C, H, W, d, d)
    patches = x2_padded.unfold(2, d, 1).unfold(3, d, 1)
    # patches: (B, C, H, W, d, d)

    # reshape for dot-product
    # x1: (B, C, H, W, 1, 1)
    x1_unsq = x1.view(B, C, H, W, 1, 1)

    # multiply and sum over channel dimension → (B, H, W, d, d)
    corr = (patches * x1_unsq).sum(dim=1)  

    # flatten the (d, d) shifts → (B, H, W, d*d)
    corr = corr.view(B, H, W, d * d)

    # move channels forward → (B, d*d, H, W)
    cost = corr.permute(0, 3, 1, 2).contiguous()
    return cost


class FeatureExtractor(nn.Module):
    """
    Multi-scale feature pyramid as in the original TF implementation.
    """
    def __init__(self):
        super().__init__()
        # conv{level}_{i}: level=1..6, i=1(stride=2),2(stride=1), plus a 3rd at level3
        self.conv1_1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(64, 96, 3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(96, 96, 3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(96, 128, 3, stride=2, padding=1)
        self.conv5_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(128, 192, 3, stride=2, padding=1)
        self.conv6_2 = nn.Conv2d(192, 192, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> dict:
        net = {}
        # Level 1
        x = lrelu(self.conv1_1(x))
        net['conv1_1'] = x
        x = lrelu(self.conv1_2(x))
        net['conv1_2'] = x
        # Level 2
        x = lrelu(self.conv2_1(x))
        net['conv2_1'] = x
        x = lrelu(self.conv2_2(x))
        net['conv2_2'] = x
        # Level 3
        x = lrelu(self.conv3_1(x))
        net['conv3_1'] = x
        x = lrelu(self.conv3_2(x))
        net['conv3_2'] = x
        x = lrelu(self.conv3_3(x))
        net['conv3_3'] = x
        # Level 4
        x = lrelu(self.conv4_1(x))
        net['conv4_1'] = x
        x = lrelu(self.conv4_2(x))
        net['conv4_2'] = x
        # Level 5
        x = lrelu(self.conv5_1(x))
        net['conv5_1'] = x
        x = lrelu(self.conv5_2(x))
        net['conv5_2'] = x
        # Level 6
        x = lrelu(self.conv6_1(x))
        net['conv6_1'] = x
        x = lrelu(self.conv6_2(x))
        net['conv6_2'] = x

        return net


class ContextNetwork(nn.Module):
    """
    Refines the flow estimate via a cascaded dilated-context module.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # in_channels = feature_channels + 2 (flow)
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(128, 96, 3, padding=8, dilation=8)
        self.conv5 = nn.Conv2d(96, 64, 3, padding=16, dilation=16)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1, dilation=1)
        # output 2-channel residual flow
        self.conv7 = nn.Conv2d(32, 2, 3, padding=1, dilation=1)

    def forward(self, feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, flow], dim=1)
        x = lrelu(self.conv1(x))
        x = lrelu(self.conv2(x))
        x = lrelu(self.conv3(x))
        x = lrelu(self.conv4(x))
        x = lrelu(self.conv5(x))
        x = lrelu(self.conv6(x))
        # residual flow
        res = self.conv7(x)
        return res


def pyramid_processing_three_frame(batch_img: torch.Tensor,
                                   feat0: dict, feat1: dict, feat2: dict,
                                   max_disp: int = 4,
                                   is_scale: bool = True) -> (torch.Tensor, torch.Tensor):
    """
    Coarse-to-fine two-way flow estimation for a triple of frames.
    Returns:
      flow_fw, flow_bw: (B, 2, H, W) at full resolution.
    """
    B, _, H_full, W_full = batch_img.shape
    device = batch_img.device
    # Initialize at the coarsest level (conv6_2)
    f6 = feat1['conv6_2']
    _, _, H6, W6 = f6.shape
    flow_fw = torch.zeros(B, 2, H6, W6, device=device)
    flow_bw = torch.zeros(B, 2, H6, W6, device=device)

    # create estimator and context modules
    # input channels to estimator = cost_volume + feature_channels + 2
    feat_ch = f6.shape[1]
    est = nn.Sequential(
        nn.Conv2d((2*max_disp+1)**2 + feat_ch + 2, 128, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 96, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(96, 64, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 32, 3, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(32, 2, 3, padding=1),
    ).to(device)

    ctx = ContextNetwork(in_channels=feat_ch + 2).to(device)

    # iterate from level 6 down to level 2
    for lvl in [6, 5, 4, 3, 2]:
        f0 = feat0[f'conv{lvl}_2']
        f1 = feat1[f'conv{lvl}_2']
        f2 = feat2[f'conv{lvl}_2']
        B, C, H, W = f1.shape

        # upsample flow from previous coarser level
        if lvl != 6:
            flow_fw = flow_resize(flow_fw, (H, W), is_scale=is_scale)
            flow_bw = flow_resize(flow_bw, (H, W), is_scale=is_scale)

        # warp neighbor features
        f2_warp = warp(f2, flow_fw)
        f0_warp = warp(f0, flow_bw)

        # cost volumes
        cv_fw = cost_volume(f1, f2_warp, max_disp)
        cv_bw = cost_volume(f1, f0_warp, max_disp)

        # estimator to predict residual flows
        input_fw = torch.cat([cv_fw, f1, flow_fw], dim=1)
        input_bw = torch.cat([cv_bw, f1, flow_bw], dim=1)
        delta_fw = est(input_fw)
        delta_bw = est(input_bw)

        flow_fw = flow_fw + delta_fw
        flow_bw = flow_bw + delta_bw

        # context refinement
        flow_fw = flow_fw + ctx(f1, flow_fw)
        flow_bw = flow_bw + ctx(f1, flow_bw)

    # finally upsample to full resolution if needed
    if H != H_full or W != W_full:
        flow_fw = F.interpolate(flow_fw, size=(H_full, W_full),
                                mode='bilinear', align_corners=True) * (float(W_full)/W)
        flow_bw = F.interpolate(flow_bw, size=(H_full, W_full),
                                mode='bilinear', align_corners=True) * (float(W_full)/W)

    return flow_fw, flow_bw


def pyramid_processing_five_frame(batch_imgs: torch.Tensor,
                                  extractor: FeatureExtractor,
                                  max_disp: int = 4,
                                  is_scale: bool = True):
    """
    Extension to five-frame input: returns three forward/backward pairs.
    """
    # split into individual frames
    img0, img1, img2, img3, img4 = torch.unbind(batch_imgs, dim=1)  # expects shape (B,5,C,H,W)
    # extract pyramids
    feat0 = extractor(img0)
    feat1 = extractor(img1)
    feat2 = extractor(img2)
    feat3 = extractor(img3)
    feat4 = extractor(img4)

    # three overlapping triples
    fw12, bw10 = pyramid_processing_three_frame(img0, feat0, feat1, feat2, max_disp, is_scale)
    fw23, bw21 = pyramid_processing_three_frame(img1, feat1, feat2, feat3, max_disp, is_scale)
    fw34, bw32 = pyramid_processing_three_frame(img2, feat2, feat3, feat4, max_disp, is_scale)

    return {
        'fw12': fw12, 'bw10': bw10,
        'fw23': fw23, 'bw21': bw21,
        'fw34': fw34, 'bw32': bw32,
    }
