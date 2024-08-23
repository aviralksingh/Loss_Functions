import torch
import torch.nn as nn
import cv2

from einops import rearrange

from segformer import *

import subprocess
import os

# Example output filename and Dropbox link
output_file = "segformer_mit_b3_cs_pretrain_19CLS_512_1024_CE_loss.pt"
dropbox_link = "https://www.dropbox.com/scl/fi/2dxmutttgtuc0qjcfbmhv/segformer_mit_b3_cs_pretrain_19CLS_512_1024_CE_loss.pt?rlkey=zeqo30tiqdve1bcbt3bzux3ru&st=team4tl1&dl=1"

# Check if the file already exists locally
if os.path.exists(output_file):
    print(f"File '{output_file}' already exists. Skipping download.")
else:
    print(f"File '{output_file}' not found. Downloading...")
    # Run the wget command to download the file
    subprocess.run(["wget", "-O", output_file, dropbox_link])


NUM_CLASSES = 19
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = segformer_mit_b3(in_channels=3, num_classes=NUM_CLASSES).to(device)


model.load_state_dict(torch.load(f'segformer_mit_b3_cs_pretrain_19CLS_512_1024_CE_loss.pt', map_location=device))


import numpy as np
import matplotlib.pyplot as plt
# utility functions to get Cityscapes Pytorch dataset and dataloaders
from utils import *

data= get_cs_datasets(rootDir='data')
sample_image, sample_label = data[0]
print(f"Input shape = {sample_image.shape}, output label shape = {sample_label.shape}")

# Check if the image was loaded successfully
if sample_image is None:
    print("Failed to load the image. Please check the path or the source.")
else:
    # Check the type and shape of the image
    print(f"Image type: {type(sample_image)}, Image shape: {sample_image.shape}")

    # Convert the image from BGR to RGB
    try:
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        print("Image successfully converted to RGB.")
    except cv2.error as e:
        print(f"Failed to convert image: {e}")
from utils import preprocess
# create torch tensor to give as input to model
pt_image = preprocess(sample_image)
pt_image = pt_image.to(device)
# get model prediction and remap certain labels to showcase
# only certain colors. class index 19 has color map (0,0,0),
# so remap unwanted classes to 19
y_pred = torch.argmax(model(pt_image.unsqueeze(0)), dim=1).squeeze(0)
predicted_labels = y_pred.cpu().detach().numpy()
# predicted_labels[(predicted_labels < 11) & (predicted_labels != 0) & (predicted_labels != 6) & (predicted_labels != 7)] = 1
# convert to corresponding color
cm_labels = (train_id_to_color[predicted_labels]).astype(np.uint8)
# overlay prediction over input frame
alpha=0.7
overlay_image = cv2.addWeighted(sample_image, 1, cm_labels, alpha, 0)
overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
predicted_output = cv2.cvtColor(cm_labels, cv2.COLOR_RGB2BGR)
cv2.imwrite("output.png", predicted_labels)


