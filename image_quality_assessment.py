
"""
Blur detection with OpenCV
https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/ -- not working well
"""
# import glob
import os
import argparse
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import random
import pyiqa
import torch

def set_random_seed(seed=123):
    """Set random seeds."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# loading variables from .env file
load_dotenv()

image_folder: str = os.getenv("IQA_FOLDER")


# IQA

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import itertools

# images_path = Path(image_folder).glob('*.jpg')  # generator
# image_path = next(images_path)
# print(image_path)

# image = Image.open(image_path)

# image_good = os.getenv("IQA_IMAGE_good")
# image_bad = os.getenv("IQA_IMAGE_bad")

# image_good = Image.open(image_good)
# image_bad = Image.open(image_bad)

# np_image_good = np.array(image_good)
# np_image_bad = np.array(image_bad)

# print(np_image)

# print(image.__str__())

# image = next(itertools.islice(images, 1))
# image = images[0]

# for image in images:
#     print(image.__str__())


transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
])

for index, image_path in enumerate(Path(image_folder).glob('*.jpg')):

    image = Image.open(image_path)

    image_np = np.array(image)

    image_tensor = transform(image=image_np)["image"]

    C, H, W = image_tensor.shape

    # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
    image_tensor = image_tensor.contiguous().view(1, C, H, W)  # (3, H, W) -> (N, 3, H, W)

    # topiq_nr / tres / clipiqa
    iqa_metric = pyiqa.create_metric('topiq_nr', device=device)

    score_fr_good = iqa_metric(image_tensor)


# img_tensor_good = transform(image=np_image_good)["image"]
# img_tensor_bad = transform(image=np_image_bad)["image"]

# print(img_tensor.shape)
# C, H, W = img_tensor_good.shape
# C_, H_, W_ = img_tensor_bad.shape

# (3, H, W) -> (N, 3, H, W)
# img_tensor_good = img_tensor_good.contiguous().view(1, C, H, W)
# img_tensor_bad = img_tensor_bad.contiguous().view(1, C_, H_, W_)
# print(img_tensor.shape)
# model_list = pyiqa.list_models()
# print((model_list == 'arniqa').any())
# print(pyiqa.list_models())

# create metric with default setting

## topiq_nr topiq_nr-flive, topiq_nr-spaq
# iqa_metric = pyiqa.create_metric('topiq_nr-spaq', device=device)  # topiq_nr / tres / clipiqa
# print(iqa_metric.seed)
# example for iqa score inference
# Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
# score_fr_good = iqa_metric(img_tensor_good)
# score_fr_bad = iqa_metric(img_tensor_bad)

# print(f'{score_fr_good=}')
# print(f'{score_fr_bad=}')

# set_random_seed(seed=195456)
# random_seed: int = random.randint(0, 2**32 - 1)

# iqa_metric = pyiqa.create_metric('topiq_nr-spaq', device=device, seed=random_seed)  #
# print(iqa_metric.seed)

# score_fr_good = iqa_metric(img_tensor_good)
# score_fr_bad = iqa_metric(img_tensor_bad)

# print(f'{score_fr_good=}')
# print(f'{score_fr_bad=}')

# blur detection
# threshold: float = 100.0
#
# def variance_of_laplacian(image):
# 	# compute the Laplacian of the image and then return the focus
# 	# measure, which is simply the variance of the Laplacian
# 	return cv2.Laplacian(image, cv2.CV_64F).var()
#
#
# # loop over the input images
# for index, imagePath in enumerate(Path(image_folder).glob('*.jpg')):
# 	# load the image, convert it to grayscale, and compute the focus measure of the image using the Variance of Laplacian
# 	# method
#
# 	# pathlib.WindowsPath to str
# 	image = cv2.imread(imagePath.__str__())
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	fm = variance_of_laplacian(gray)
# 	text = "Not Blurry"
# 	# if the focus measure is less than the supplied threshold,
# 	# then the image should be considered "blurry"
# 	if fm < threshold:
# 		text = "Blurry"
# 	# show the image
# 	cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
#
# 	cv2.imshow("Image", image)
# 	key = cv2.waitKey(0)