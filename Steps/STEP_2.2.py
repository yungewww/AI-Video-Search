"""
STEP 2.2
Input:
- Screenshots
Output:
- cropped_images.csv: [vidId, frameNum, timestamp, detectedObjId, detectedObjClass, confidence, bbox info]
- cropped_images
- bbox_images
"""

import torch, torchvision
import numpy as np
import os, csv, cv2, random
from pathlib import Path

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
setup_logger()

# Set Up Predictor
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# CREATE PATHS
cropped_images_dir = "./cropped_images"
os.makedirs(cropped_images_dir, exist_ok=True)

bbox_images_dir = "./bbox_images"
os.makedirs(bbox_images_dir, exist_ok=True)

csv_file = "cropped_images.csv"
fieldnames = ['vidId', 'frameNum', 'timestamp', 'detectedObjId', 'detectedObjClass', 'confidence', 'bbox_info']

screenshots_folder = os.listdir("screenshots")

# START
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # FOR EACH SCREENSHOT
    for image in screenshots_folder:
        image_path = os.path.join("screenshots", image)
        im = cv2.imread(image_path)
        if im is None:
            print(f"Could not load image from {image_path}")
            continue

        # GET IMAGE NAME & INFO
        filename = Path(image_path).stem
        vidID, frameNum, timestamp = filename.split("_")

        # PREDICT
        outputs = predictor(im)

        # 1. SAVE BBOX IMAGES
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        visualized_image = out.get_image()[:, :, ::-1]
        bbox_save_path = os.path.join(bbox_images_dir, f"{vidID}_{frameNum}_bbox.jpg")  # 指定文件名
        cv2.imwrite(bbox_save_path, visualized_image)

        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()

        for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            # 2. SAVE CROPPED IMAGES
            x1, y1, x2, y2 = [int(i) for i in box]
            cropped_image = im[y1:y2, x1:x2]
            cropped_save_path = os.path.join(cropped_images_dir, f"{vidID}_{frameNum}_{idx}.jpg")
            cv2.imwrite(cropped_save_path, cropped_image)

            # 3. SAVE AS CSV
            bbox_info = f"{x1},{y1},{x2},{y2}"
            class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", [])[cls]

            writer.writerow({
                'vidId': vidID,
                'frameNum': frameNum,
                'timestamp': timestamp,
                'detectedObjId': idx,
                'detectedObjClass': class_name,
                'confidence': score,
                'bbox_info': bbox_info
            })
            print(f"{vidID}, {frameNum}, {timestamp}, {idx}, {class_name}, {score}, {bbox_info}")