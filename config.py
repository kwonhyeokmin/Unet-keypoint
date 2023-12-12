import os
import os.path as osp
import argparse
from common.utils.dir_utils import make_folder
import albumentations as A
import yaml
from easydict import EasyDict as edict


class Config:
    # directories
    cur_dir = osp.dirname(os.path.abspath(__file__))

    root_dir = cur_dir
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')

    batch_size = 32
    num_thread = 16
    CROP_IMG_HEIGHT = 224
    CROP_IMG_WIDTH = 224

    # augmentation
    data_transforms = {
        "train": A.Compose([
            # A.CenterCrop(CROP_IMG_HEIGHT, CROP_IMG_WIDTH),
            A.Resize(CROP_IMG_HEIGHT, CROP_IMG_WIDTH),
            A.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            A.CoarseDropout(max_holes=8, max_height=CROP_IMG_HEIGHT // 20, max_width=CROP_IMG_WIDTH // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ]),

        "valid": A.Compose([
            A.Resize(CROP_IMG_HEIGHT, CROP_IMG_WIDTH),
            # A.CenterCrop(CROP_IMG_HEIGHT, CROP_IMG_WIDTH),
        ]),

        "test": A.Compose([
            A.Resize(CROP_IMG_HEIGHT, CROP_IMG_WIDTH),
        ])
    }


cfg = Config()

def update_config(config_file):
    # Make folder
    make_folder(cfg.output_dir)
    make_folder(cfg.model_dir)
    make_folder(cfg.vis_dir)

    with open(config_file) as f:
        updated_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    cfg.hyp = updated_config
