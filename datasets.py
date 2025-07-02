# datasets.py

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from utils import mvn

class BasicDataset(Dataset):
    """
    테스트/추론용 데이터셋.
    data_list_file: 각 줄에 img0, img1, img2, save_name 순으로 파일명 나열된 txt
    img_dir: 이미지 파일이 들어 있는 디렉토리
    is_normalize_img: mvn(mean–variance normalization) 적용 여부
    """
    def __init__(self,
                 data_list_file: str,
                 img_dir: str,
                 is_normalize_img: bool = True):
        self.img_dir = img_dir
        # load as str array; 각 행(row)이 ['f0.png','f1.png','f2.png','seq1'] 형태
        self.data_list = np.loadtxt(data_list_file, dtype=str)
        if self.data_list.ndim == 1:
            # 파일에 한 줄만 있는 경우도 (1,4) 형태로 맞춰준다
            self.data_list = self.data_list[np.newaxis, :]
        self.is_normalize_img = is_normalize_img

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        row = self.data_list[idx]
        # 첫 3개 원소는 프레임 이미지 파일명
        img_files = row[:3]
        imgs = []
        for fname in img_files:
            path = os.path.join(self.img_dir, fname)
            img = Image.open(path).convert('RGB')
            tensor = TF.to_tensor(img)            # [0,1]
            if self.is_normalize_img:
                tensor = mvn(tensor)               # 채널별 MVN
            imgs.append(tensor)
        # shape: (3, C, H, W)
        imgs = torch.stack(imgs, dim=0)

        # 마지막 컬럼은 저장 시 사용할 이름 (e.g. 'seq1')
        save_name = row[3] if row.shape[0] > 3 else None

        return imgs, save_name


import numpy as np
from flowlib import read_flo
from data_augmentation import random_flip_with_flow, flow_resize, random_crop

class TrainDataset(torch.utils.data.Dataset):
    """
    3-프레임 입력 + 양방향 GT flow + occlusion mask 로드.
    data_list_file: 각 줄에 img0, img1, img2, seq_name
    img_dir: 이미지 디렉토리
    flow_fw_dir, flow_bw_dir: .flo GT flow 디렉토리 (flow_fw_dir/seq_name.flo)
    crop_size: 학습 시 random crop 크기
    augment: 플립·크롭 증강 사용 여부
    """
    def __init__(self,
                 data_list_file: str,
                 img_dir: str,
                 flow_fw_dir: str,
                 flow_bw_dir: str,
                 crop_size=(256,256),
                 augment=True):
        self.data = np.loadtxt(data_list_file, dtype=str)
        if self.data.ndim == 1:
            self.data = self.data[np.newaxis, :]
        self.img_dir = img_dir
        self.flow_fw_dir = flow_fw_dir
        self.flow_bw_dir = flow_bw_dir
        self.crop_h, self.crop_w = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img0_f, img1_f, img2_f, seq = self.data[idx]
        # load images
        imgs = []
        for fn in (img0_f, img1_f, img2_f):
            im = Image.open(os.path.join(self.img_dir, fn)).convert('RGB')
            t = TF.to_tensor(im)   # [0,1]
            t = mvn(t)             # 채널별 MVN
            imgs.append(t)
        # load GT flows
        fw = torch.from_numpy(read_flo(os.path.join(self.flow_fw_dir, seq + '.flo'))).permute(2,0,1)
        bw = torch.from_numpy(read_flo(os.path.join(self.flow_bw_dir, seq + '.flo'))).permute(2,0,1)
        # occlusion mask (non-occluded) 계산
        mask_noc, _ = occlusion(fw.unsqueeze(0), bw.unsqueeze(0))
        mask_noc = mask_noc[0]  # (1,H,W)

        # 증강
        if self.augment:
            # random flip with flow
            imgs, flows = random_flip_with_flow(imgs, [fw, bw])
            fw, bw = flows
            # random crop
            imgs = random_crop(imgs, self.crop_h, self.crop_w)
            fw, bw = random_crop([fw, bw], self.crop_h, self.crop_w)
            mask_noc = random_crop([mask_noc], self.crop_h, self.crop_w)[0]

        # resize to be divisible by 64 (for pyramid)
        H, W = imgs[0].shape[-2:]
        H64 = ((H+63)//64)*64
        W64 = ((W+63)//64)*64
        imgs = [F.interpolate(t.unsqueeze(0), size=(H64, W64),
                              mode='bilinear', align_corners=True)[0] for t in imgs]
        fw = flow_resize(fw, (H64, W64))
        bw = flow_resize(bw, (H64, W64))
        mask_noc = F.interpolate(mask_noc.unsqueeze(0).unsqueeze(0),
                                  size=(H64, W64),
                                  mode='nearest')[0]

        # stack
        imgs = torch.stack(imgs, dim=0)         # (3, C, H64, W64)
        target = torch.stack([fw, bw, mask_noc], dim=0)
        # (3, 2 or 1, H64, W64) – caller에서 분리 사용

        return imgs, target