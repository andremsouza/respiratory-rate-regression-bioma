"""Semantic segmentation test with PyTorch."""

# %% [markdown]
# # Imports

# %%
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import make_grid
from torchvision.io.image import read_image
from torchvision.io.video import read_video
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

# %%

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


# %%

dog1_int = read_image(str(Path("images") / "dog1.jpg"))
dog2_int = read_image(str(Path("images") / "dog2.jpg"))
dog_list = [dog1_int, dog2_int]

grid = make_grid(dog_list)
show(grid)

# %%
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430]], dtype=torch.float)
colors = ["blue", "yellow"]
result = draw_bounding_boxes(dog1_int, boxes, colors=colors, width=5)
show(result)

# %%


weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

images = [transforms(d) for d in dog_list]

model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
model = model.eval()

outputs = model(images)
print(outputs)

# %%
score_threshold = 0.8
dogs_with_boxes = [
    draw_bounding_boxes(
        dog_int, boxes=output["boxes"][output["scores"] > score_threshold], width=4
    )
    for dog_int, output in zip(dog_list, outputs)
]
show(dogs_with_boxes)

# %%

weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

images = [transforms(d) for d in dog_list]

model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
model = model.eval()

output = model(images)
print(output)

# %%
dog1_output = output[0]
dog1_masks = dog1_output["masks"]
print(
    f"shape = {dog1_masks.shape}, dtype = {dog1_masks.dtype}, "
    f"min = {dog1_masks.min()}, max = {dog1_masks.max()}"
)

# %%
print("For the first dog, the following instances were detected:")
print([weights.meta["categories"][label] for label in dog1_output["labels"]])

# %%
proba_threshold = 0.5
dog1_bool_masks = dog1_output["masks"] > proba_threshold
print(f"shape = {dog1_bool_masks.shape}, dtype = {dog1_bool_masks.dtype}")

# There's an extra dimension (1) to the masks. We need to remove it
dog1_bool_masks = dog1_bool_masks.squeeze(1)

show(draw_segmentation_masks(dog1_int, dog1_bool_masks, alpha=0.9))

# %%
print(dog1_output["scores"])

# %%
score_threshold = 0.75

boolean_masks = [
    out["masks"][out["scores"] > score_threshold] > proba_threshold for out in output
]

dogs_with_masks = [
    draw_segmentation_masks(img, mask.squeeze(1))
    for img, mask in zip(dog_list, boolean_masks)
]
show(dogs_with_masks)

# %% [markdown]
# # Test on videos

# %%
video, audio, info = read_video("videos/video1_01.mp4")
# Take a frame from the video
frame = video[0]

# %%
show(frame.permute(2, 0, 1))

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
transforms = weights.transforms()

model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False)
model = model.eval().to(DEVICE)

# %%
frame = transforms(frame.permute(2, 0, 1)).to(DEVICE)
output = model([frame])
print(output)

# %%
cow_output = output[0]
cow_masks = cow_output["masks"]
print(
    f"shape = {cow_masks.shape}, dtype = {cow_masks.dtype}"
    f"min = {cow_masks.min()}, max = {cow_masks.max()}"
)


# %%
print("For the first cow, the following instances were detected:")
print([weights.meta["categories"][label] for label in cow_output["labels"]])

# %%
proba_threshold = 0.5
cow_bool_masks = cow_output["masks"] > proba_threshold
print(f"shape = {cow_bool_masks.shape}, dtype = {cow_bool_masks.dtype}")

# There's an extra dimension (1) to the masks. We need to remove it
cow_bool_masks = cow_bool_masks.squeeze(1)

# Convert frame to uint8
frame_uint8 = frame.type(torch.uint8).cpu()

show(draw_segmentation_masks(frame_uint8, cow_bool_masks, alpha=0.9))

# %%
print(cow_output["scores"])

# %%
score_threshold = 0.5

boolean_masks = [
    out["masks"][out["scores"] > score_threshold] > proba_threshold for out in output
]

cows_with_masks = [draw_segmentation_masks(frame_uint8, boolean_masks[0].squeeze(1))]
show(cows_with_masks)

# %%
