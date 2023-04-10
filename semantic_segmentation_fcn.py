"""Semantic segmentation test with PyTorch."""

# %% [markdown]
# # Imports

# %%
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io.image import read_image
from torchvision.io.video import read_video
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image

# %% [markdown]
# # Constants

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# %% [markdown]
# # Test

# %%
img = read_image("images/dog1.jpg")

# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).to(DEVICE)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0).to(DEVICE)

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, class_to_idx["dog"]]
# Save the mask as a grayscale image
to_pil_image(mask).save("dog_mask_fcn.png")

# %% [markdown]
# # Test on videos

# %%
video, audio, info = read_video("videos/video1_01.mp4")
# Take a frame from the video
frame = video[0]

# %%

# Step 3: Apply inference preprocessing transforms
batch = preprocess(frame.permute(2, 0, 1)).unsqueeze(0).to(DEVICE)

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
mask = normalized_masks[0, class_to_idx["cow"]]
# Save the mask as a grayscale image
to_pil_image(mask).save("cow_mask_fcn.png")

# %%
# Apply process to all frame of the video
# Batches of size 1 are used to avoid memory issues

for i in range(0, len(video), 1):
    batch = preprocess(video[i].permute(2, 0, 1)).unsqueeze(0).to(DEVICE)
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    mask = normalized_masks[0, class_to_idx["cow"]]
    # Save the mask of each frame as a grayscale image
    to_pil_image(mask).save(f"mask_fcn/cow_mask_{i}.png")

# %%
# Produce video with the masks
# grayscale images are used as masks

# Read the first frame
img = cv2.imread("mask_fcn/cow_mask_0.png")
height, width, layers = img.shape

# get number of frames
len_video = len(os.listdir("mask_fcn/"))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("cow_mask_fcn.mp4", fourcc, 24.0, (width, height))

for i in range(len_video):
    img = cv2.imread(f"mask_fcn/cow_mask_{i}.png")
    out.write(img)

out.release()


# %%
