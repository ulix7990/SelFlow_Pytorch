# selflow_model.py

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2


import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from datasets import TrainDataset


from network import FeatureExtractor, pyramid_processing_three_frame  # PyTorch 네트워크 모듈들
from datasets import BasicDataset                                 # PyTorch 데이터셋
from data_augmentation import flow_resize                         # 크기 보정 유틸
from flowlib import flow_to_color, write_flo                      # .flo I/O 및 시각화
from utils import lrelu, occlusion                                # 활성화·occlusion 마스크

class SelFlowModel:
    def __init__(self,
                 batch_size=8,
                 is_scale=True,
                 num_input_threads=4,
                 save_dir='KITTI',
                 checkpoint_dir='checkpoints',
                 model_name='model',
                 is_restore_model=False,
                 restore_model_path='',
                 dataset_config=None):
        """
        TensorFlow 버전 :contentReference[oaicite:0]{index=0} 을 참고하여 포팅.
        - batch_size: 테스트 시 1로 고정해도 무방합니다.
        - is_scale: 피라미드 스케일링 여부.
        - num_input_threads: DataLoader num_workers.
        """
        self.batch_size = batch_size
        self.is_scale = is_scale
        self.num_workers = num_input_threads

        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델(FeatureExtractor + estimator/context 모듈은 내부에서 동적 생성)
        self.extractor = FeatureExtractor().to(self.device)

        # 디렉토리 준비
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.save_dir, checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model_name = model_name

        self.is_restore = is_restore_model
        self.ckpt_path = restore_model_path
        if self.is_restore:
            self._load_checkpoint(self.ckpt_path)

        # 데이터 설정
        assert dataset_config is not None, "dataset_config must be provided"
        self.dataset_config = dataset_config


    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        # extractor만 저장했다고 가정
        self.extractor.load_state_dict(checkpoint['extractor'])
        print(f"Checkpoint loaded from {path}")


    def test(self, restore_model: str, save_dir: str, is_normalize_img: bool = True):
        """
        테스트 모드: 각 시퀀스별로 flow_fw, flow_bw 및 컬러 이미지를 저장.
        TensorFlow 버전 :contentReference[oaicite:1]{index=1} 을 참고.
        """
        # 데이터로더 준비
        dataset = BasicDataset(
            data_list_file=self.dataset_config['data_list_file'],
            img_dir=self.dataset_config['img_dir'],
            is_normalize_img=is_normalize_img
        )
        loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=self.num_workers)

        # 체크포인트 로드
        if restore_model:
            self._load_checkpoint(restore_model)

        os.makedirs(save_dir, exist_ok=True)

        self.extractor.eval()
        with torch.no_grad():
            for idx, (imgs, save_name) in enumerate(loader):
                # imgs: (1, 3, C, H, W) → unpack
                # BasicDataset 에서 (T=3, C, H, W) 스택했으므로
                # imgs.shape == (1, 3, C, H, W)
                seq = imgs[0]  # (3, C, H, W)
                img0, img1, img2 = [frame.to(self.device) for frame in seq]

                _, C, H, W = img1.shape

                # 64의 배수로 패딩/리사이즈
                new_H = ((H + 63) // 64) * 64
                new_W = ((W + 63) // 64) * 64
                img0r = F.interpolate(img0.unsqueeze(0), size=(new_H, new_W),
                                      mode='bilinear', align_corners=True)
                img1r = F.interpolate(img1.unsqueeze(0), size=(new_H, new_W),
                                      mode='bilinear', align_corners=True)
                img2r = F.interpolate(img2.unsqueeze(0), size=(new_H, new_W),
                                      mode='bilinear', align_corners=True)

                # 피라미드 처리 (forward/backward flow)
                flow_fw, flow_bw = pyramid_processing_three_frame(
                    torch.cat([img0r, img1r, img2r], dim=0).unsqueeze(0),
                    # 내부에서 FeatureExtractor 호출하므로 extractor는 사용하지 않고도 동작
                    None, None, None,
                    max_disp=4, is_scale=self.is_scale
                )

                # 원래 크기로 리사이즈
                flow_fw = flow_resize(flow_fw, (H, W), is_scale=self.is_scale)[0]
                flow_bw = flow_resize(flow_bw, (H, W), is_scale=self.is_scale)[0]

                # 컬러 맵 생성
                fw_color = (flow_to_color(flow_fw.cpu().numpy()) * 255).astype(np.uint8)
                bw_color = (flow_to_color(flow_bw.cpu().numpy()) * 255).astype(np.uint8)

                # 저장
                cv2.imwrite(f"{save_dir}/flow_fw_color_{save_name[0]}.png", cv2.cvtColor(fw_color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{save_dir}/flow_bw_color_{save_name[0]}.png", cv2.cvtColor(bw_color, cv2.COLOR_RGB2BGR))
                write_flo(f"{save_dir}/flow_fw_{save_name[0]}.flo", flow_fw.cpu().numpy())
                write_flo(f"{save_dir}/flow_bw_{save_name[0]}.flo", flow_bw.cpu().numpy())

                print(f"[{idx+1}/{len(loader)}] saved {save_name[0]}")

    def train(self,
              train_list_file: str,
              img_dir: str,
              flow_fw_dir: str,
              flow_bw_dir: str,
              epochs: int = 30,
              lr: float = 1e-4,
              weight_decay: float = 1e-5,
              save_interval: int = 5):
        """
        학습 루프 구현.
        - train_list_file: 학습용 리스트(txt)
        - img_dir, flow_*_dir: 데이터 경로
        - epochs: 전체 에폭 수
        - lr, weight_decay: Adam 하이퍼파라미터
        - save_interval: 체크포인트 저장 주기(에폭)
        """
        # 1) Dataset & DataLoader
        train_ds = TrainDataset(
            data_list_file=train_list_file,
            img_dir=img_dir,
            flow_fw_dir=flow_fw_dir,
            flow_bw_dir=flow_bw_dir,
            crop_size=(256,256),
            augment=True
        )
        loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # 2) Optimizer & Scheduler
        params = list(self.extractor.parameters())
        # TODO: pyramid 처리 모듈(estimator, context) 파라미터도 함께 추가
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        # 3) 학습 루프
        for epoch in range(1, epochs+1):
            self.extractor.train()
            total_loss = 0.0
            for imgs, target in loader:
                # imgs: (B,3,C,H64,W64)
                # target: (B,3,?,H64,W64) -> [fw, bw, mask_noc]
                imgs = imgs.to(self.device)
                fw_gt, bw_gt, mask = target[:,0:2], target[:,2:3], None
                fw_gt = fw_gt.to(self.device)
                bw_gt = bw_gt.to(self.device)
                mask = target[:,2:3].to(self.device)

                optimizer.zero_grad()
                # 피라미드 처리
                # FeatureExtractor로 피처 추출
                img0, img1, img2 = imgs[:,0], imgs[:,1], imgs[:,2]
                feat0 = self.extractor(img0)
                feat1 = self.extractor(img1)
                feat2 = self.extractor(img2)
                # forward/backward 추정
                fw_pred, bw_pred = pyramid_processing_three_frame(
                    torch.cat([img0, img1, img2], dim=1),
                    feat0, feat1, feat2,
                    max_disp=4, is_scale=self.is_scale
                )
                # GT와 동일 크기로 resize (이미 64 배수)
                # mask_noc: (B,1,H64,W64)
                # L1 loss masked
                loss_fw = F.l1_loss(fw_pred * mask, fw_gt * mask, reduction='mean')
                loss_bw = F.l1_loss(bw_pred * mask, bw_gt * mask, reduction='mean')
                loss = loss_fw + loss_bw

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * imgs.size(0)

            scheduler.step()
            avg_loss = total_loss / len(train_ds)
            print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

            # 체크포인트 저장
            if epoch % save_interval == 0 or epoch == epochs:
                ckpt = {
                    'extractor': self.extractor.state_dict(),
                    # TODO: estimator/context 모듈 state_dict도 포함
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                path = os.path.join(self.checkpoint_dir,
                                    f"{self.model_name}_epoch{epoch}.pth")
                torch.save(ckpt, path)
                print(f"Saved checkpoint: {path}")

