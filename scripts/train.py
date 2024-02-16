import argparse
import os
import sys

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.optim import lr_scheduler

sys.path.insert(0, os.getcwd())
from scripts.config import get_train_cfg, get_val_cfg
from scripts.dataset import get_coco_dataset
from scripts.loss import mask_rcnn_loss


from train_epoch import (
    collect_samples,
    loss_per_epoch,
    model_checkpoint,
    train_epoch,
)


def main(
    save_folder_name,
):
    # cfg
    train_cfg = get_train_cfg(
        folder_name=save_folder_name,
    )
    val_cfg = get_val_cfg(
        folder_name=save_folder_name,
    )

    model = maskrcnn_resnet50_fpn(pretrained=False).to(train_cfg.DEVICE)

    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg.LEARNING_RATE,
        momentum=train_cfg.MOMENTUM,
        weight_decay=train_cfg.WEIGHT_DECAY,
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=train_cfg.GAMMA)

    # Load dataset
    dataset_train = get_coco_dataset(train_cfg)
    dataset_val = get_coco_dataset(val_cfg)

    # select some sequences to preview
    list_samples = collect_samples(dataset_val)

    print(
        f"The size of training dataset is {len(dataset_train)} and the size of val dataset is {len(dataset_val)}"
    )

    train_loss_list = []
    val_loss_list = []

    for epoch in range(train_cfg.EPOCHS):
        model_checkpoint(model, epoch, list_samples, train_cfg)

        if epoch != 0:
            loss_per_epoch(train_loss_list, val_loss_list, epoch, train_cfg)

        train_loss, val_loss = train_epoch(
            dataset_train,
            dataset_val,
            model,
            optimizer,
            mask_rcnn_loss,
            epoch,
            train_cfg,
            scheduler,
        )

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument("--save_folder_name", type=str, default="test")

    # Parse the command-line arguments
    args = parser.parse_args()

    main(
        save_folder_name=args.save_folder_name,
    )
