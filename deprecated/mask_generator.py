"""
Deprecated: find out you don't need to detect and inpaint just do the cropping
but been doing this quite sometime so keeping this code

Generating mask(s) for inpainting methods
"""


import cv2
import numpy as np

# Define the image dimensions and the coordinates for the annotation
width, height = 1920, 1080
minx, maxx, miny, maxy = 5, 764, 5, 27

# Create a black image (mask)
mask = np.zeros((height, width), dtype=np.uint8)

# Draw a white rectangle on the mask to represent the annotation region
cv2.rectangle(mask, (minx, miny), (maxx, maxy), color=255, thickness=-1)

# Save the mask to a file
cv2.imwrite('mask/mask_1920_1080_TL.png', mask)

# Display the mask (optional)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()