import os
import sys
import warnings

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn


warnings.simplefilter("ignore", UserWarning)

sys.path.insert(0, os.getcwd())
from scripts.eval import calculate_mAP


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        m.bias.data.fill_(0.01)


def train_epoch(
    train_dataset, val_dataset, model, optimizer, loss_fn, epoch, cfg, scheduler
):
    # forward pass
    loop = tqdm(train_dataset, leave=True)
    mean_loss = []
    val_loss = []
    print(f"epoch: {epoch+1} out of {cfg.EPOCHS}")

    model.train()
    for batch_idx, (x_batch, y_batch) in enumerate(loop):

        optimizer.zero_grad()
        # forward pass
        x_batch = x_batch.float().to(cfg.DEVICE)

        for key, _ in y_batch.items():
            y_batch[key] = y_batch[key].to(cfg.DEVICE)

        loss_values = model([x_batch], [y_batch])

        loss = loss_fn(loss_values, cfg)

        # back prop
        mean_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx == len(train_dataset) - 1:
            for i in range(len(val_dataset)):
                x_batch, y_batch = val_dataset[i]
                x_batch = x_batch.float().to(cfg.DEVICE)
                for key, _ in y_batch.items():
                    y_batch[key] = y_batch[key].to(cfg.DEVICE)
                # validation
                with torch.no_grad():
                    loss_values = model([x_batch], [y_batch])
                    loss_val = loss_fn(loss_values, cfg)
                value_loss = loss_val.item()
                val_loss.append(value_loss)

            loop.set_postfix(
                train_loss=np.mean(mean_loss),
                val_loss=np.mean(val_loss),
            )
            break
    
        else:
            # update progress bar
            loop.set_postfix(train_loss=np.mean(mean_loss), val_loss="TBD")

    if cfg.USE_SCHEDULER:
        scheduler.step()

    return np.mean(mean_loss), np.mean(val_loss)


def model_checkpoint(model, epoch, list_samples, cfg, mtype="model"):
    """This function checks the model and saves the model weights and shows some image inferences.

    Args:
        model ([torch model]): The torch deep learning model
        epoch ([int]): The current epoch
        list_of_images ([list]): List of images for inference and preview
        list_of_ground_truth ([List]): List of labels for inference and preview
    """
    check_path = cfg.ENS_CHECK_PATH if mtype == "ens" else cfg.CHECK_PATH

    # save the model
    check_point_name = "epoch_" + str(epoch) + "_model.pt"
    save_path = os.path.join(check_path, check_point_name)
    output_save = open(save_path, mode="wb")
    torch.save(model.state_dict(), output_save)

    for sample_idx, sample in enumerate(list_samples):
        get_sample_results(sample, model, cfg, epoch, sample_idx)


def loss_per_epoch(train_loss_list, val_loss_list, epoch, cfg):
    """This function plots loss per epoch for train and validation.

    Args:
        train_loss_list ([list]): list of train losses.
        val_loss_list ([type]):  list of val losses.
        epoch ([int]): epoch number for plot name
        check_path ([str]): the path to save the figure
        ID (str, optional): The unique id for the name of the figure. Defaults to "unet32_".
    """
    fig, axs = plt.subplots(1, figsize=(6, 6))

    # Plot loss per epoch
    axs.plot(list(range(0, epoch)), train_loss_list, "b")
    axs.plot(list(range(0, epoch)), val_loss_list, "r")
    axs.legend(["train loss", "validation loss"])
    axs.set_title("Loss per epoch")
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Loss")

    plt.savefig(cfg.CHECK_PATH + "/loss_per_epoch.png")


def collect_samples(dataset_val):
    samples = [500, 700, 800, 1700, 1800]
    list_of_data = []

    for item in samples:
        data = dataset_val[item]
        list_of_data.append(data)

    return list_of_data


def get_sample_results(sample, model, cfg, epoch, sample_idx):
    image, _ = sample

    model.eval()
    model.cuda()

    image_tensor = image.unsqueeze(0).cuda()

    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]["boxes"].cpu().numpy()
    labels = predictions[0]["labels"].cpu().numpy()
    masks = predictions[0]["masks"].cpu().numpy()

    _, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(np.transpose(image, (1, 2, 0)))

    for box, label, mask in zip(boxes, labels, masks):
        x, y, w, h = box
        rect = patches.Rectangle(
            (x, y), w - x, h - y, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax[0].add_patch(rect)
        ax[0].text(x, y, f"Label: {label}", color="red")
        mask = np.transpose(mask, (1, 2, 0))
        ax[1].imshow(mask[:, :, 0], alpha=0.1, cmap="Reds")

    save_path = (
        cfg.CHECK_PATH
        + "/_epoch"
        + str(epoch)
        + "__sample__"
        + str(sample_idx)
        + ".png"
    )
    plt.savefig(save_path)
    plt.close()
