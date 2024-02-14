import torchvision


if __name__ == "__main__":
    import torch
    import torchvision.transforms as transforms
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torchvision import transforms as T

    print("PyTorch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)

    # Load pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Define the transformation to apply to the input image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Load and preprocess the input image
    image_path = (
        r"C:\Projects\maskRCNN_object_detection\.coco_dataset\images\000000001580.jpg"
    )
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    print(image_tensor.shape)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    print("predictions[0]['boxes'].shape: ", predictions[0]["boxes"].shape)
    print("predictions[0]['labels'].shape: ", predictions[0]["labels"].shape)
    print("predictions[0]['masks'].shape: ", predictions[0]["masks"].shape)

    # Post-process predictions
    # For example, print predicted boxes, labels, and masks
    boxes = predictions[0]["boxes"].numpy()
    labels = predictions[0]["labels"].numpy()
    masks = predictions[0]["masks"].numpy()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(image)

    for box, label, mask in zip(boxes, labels, masks):
        x, y, w, h = box
        rect = patches.Rectangle(
            (x, y), w - x, h - y, linewidth=1, edgecolor="r", facecolor="none"
        )
        ax[0].add_patch(rect)
        ax[0].text(x, y, f"Label: {label}", color="red")
        mask = np.transpose(mask, (1, 2, 0))
        ax[1].imshow(mask[:, :, 0], alpha=0.1, cmap="Reds")

    plt.show()
