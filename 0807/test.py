import os
import numpy as np
import cv2

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets.builder import DATASETS

from mmcv import Config
from mmdet.datasets.coco import CocoDataset


config_file = "./configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py"

cfg = Config.fromfile(config_file)

@DATASETS.register_module(force=True)

class ParkingDataset(CocoDataset):
    CLASSES = ('세단(승용차)', 'SUV', '승합차', '버스','학원차량(통학버스)',
           '트럭','택시','성인','어린이','오토바이','전동킥보드','자전거','유모차','쇼핑카트')

d_type = "ParkingDataset"
cfg.dataset_type = 'ParkingDataset'
cfg.data_root = "./Data/"


cfg.data.train.type = "ParkingDataset"
cfg.data.train.ann_file = "./Data/train.json"
cfg.data.train.img_prefix = "./Data/images/"

cfg.data.val.type = d_type
cfg.data.val.ann_file = "./Data/valid.json"
cfg.data.val.img_prefix = "./Data/images/"

cfg.data.test.type = d_type
cfg.data.test.ann_file = "./Data/test.json"
cfg.data.test.img_prefix = "./Data/images/"

cfg.model.roi_head.bbox_head.num_classes = 14
cfg.load_from = "./epoch_15.pth"


#train checkpoint save dir path
cfg.work_dir = "./work_dirs/0804"

cfg.lr_config.warmup = None
cfg.log_config.interval = 10

cfg.evaluation.metric = 'bbox'
cfg.evaluation.interval = 6
cfg.checkpoint_config.interval = 6

cfg.runner.max_epochs = 10
cfg.seed = 0
cfg.gpu_ids = range(1)
set_random_seed(0, deterministic=False)

with open("./Data/test.json", 'r', encoding='utf-8') as f:
    image_infos = json.loads(f.read())


score_threshold = 0.5

for img_info in image_infos['images']:
    file_name = img_info['file_name']
    image_path = os.path.join("./Data/images/", file_name)

    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow("Test", img)
    cv2.waitKey(0)

    exit()