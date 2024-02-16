import os
import json

import ml_collections
import torch
from torchvision import transforms


def get_train_cfg(folder_name="Test"):
    config = ml_collections.ConfigDict()

    config.BEST_MODEL_PATH = None

    config.CHECK_PATH = f"./.results/{folder_name}"

    if not os.path.isdir(config.CHECK_PATH):
        os.mkdir(config.CHECK_PATH)

    config.DATA_DIRECTORY = "./.coco_dataset/instances_train2017.json"

    # PARAMETERS
    config.LEARNING_RATE = 0.001
    config.WEIGHT_DECAY = 0.000000001
    config.EPOCHS = 300
    config.PIN_MEMORY = True
    config.MOMENTUM = 0.9
    config.USE_SCHEDULER = True
    config.GAMMA = 0.9

    # debug
    config.DEBUG_TRAIN = True

    config.SHAPE = (512, 512)

    # device
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    config.transforms_image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
        ]
    )

    # Compute the total loss
    config.loss_classifier = 1.0
    config.loss_box_reg = 5.0
    config.loss_mask = 2.0
    config.loss_objectness = 1.0
    config.loss_rpn_box_reg = 5.0

    # MAP LIST
    config.mAP_list = []

    save_hyperparams(config, "cfgtrain.json")

    return config


def get_val_cfg(folder_name="Test"):

    config = get_train_cfg(folder_name)

    config.DATA_DIRECTORY = "./.coco_dataset/instances_val2017.json"

    # transforms
    config.transforms_image = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    save_hyperparams(config, "cfgval.json")

    return config


def save_hyperparams(config, file_name):
    cfg_json = config.to_json_best_effort()
    json_path = os.path.join(config.CHECK_PATH, file_name)
    with open(json_path, "w") as outfile:
        json.dump(cfg_json, outfile)
