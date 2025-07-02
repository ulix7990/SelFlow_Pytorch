# main.py

import os
import argparse

from extract_config import config_dict
from selflow_model import SelFlowModel

def main():
    parser = argparse.ArgumentParser(description='SelFlow PyTorch runner')
    parser.add_argument('--config', type=str, default='./config/config.ini',
                        help='Path to the INI config file')
    args = parser.parse_args()

    # 1) Load config
    config = config_dict(args.config)
    run_cfg     = config.get('run', {})
    dataset_cfg = config.get('dataset', {})

    # 2) GPU selection (optional)
    if 'cuda_visible_devices' in run_cfg:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(run_cfg['cuda_visible_devices'])

    # 3) Instantiate model
    model = SelFlowModel(
        batch_size=        run_cfg.get('batch_size', 1),
        is_scale=          run_cfg.get('is_scale', True),
        num_input_threads=run_cfg.get('num_input_threads', 4),
        save_dir=          run_cfg.get('save_dir', 'results'),
        checkpoint_dir=    run_cfg.get('checkpoint_dir', 'checkpoints'),
        model_name=        run_cfg.get('model_name', 'model'),
        is_restore_model=  run_cfg.get('is_restore_model', False),
        restore_model_path=run_cfg.get('restore_model', ''),
        dataset_config=    dataset_cfg
    )

    # 4) Run
    mode = run_cfg.get('mode', 'test').lower()
    if mode == 'test':
        model.test(
            restore_model=   run_cfg.get('restore_model', ''),
            save_dir=        run_cfg.get('save_dir', 'results'),
            is_normalize_img=dataset_cfg.get('is_normalize_img', True)
        )

    elif mode == 'train':
        # 필수로 config.ini 의 [dataset] 섹션에 다음 필드를 추가해야 합니다.
        #   train_list_file, img_dir, flow_fw_dir, flow_bw_dir
        model.train(
            train_list_file=dataset_cfg['train_list_file'],
            img_dir=         dataset_cfg['img_dir'],
            flow_fw_dir=     dataset_cfg['flow_fw_dir'],
            flow_bw_dir=     dataset_cfg['flow_bw_dir'],
            epochs=          run_cfg.get('epochs', 30),
            lr=              run_cfg.get('lr', 1e-4),
            weight_decay=    run_cfg.get('weight_decay', 1e-5),
            save_interval=   run_cfg.get('save_interval', 5)
        )

    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'train' or 'test'.")

if __name__ == '__main__':
    main()
