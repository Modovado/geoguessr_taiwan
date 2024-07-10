"""
Deprecated: find out you don't need to detect and inpaint just do the cropping
but been doing this quite sometime so keeping this code

Watermark detector:
To detect whether the folder of photos or images contains watermark(yellow).

analyze_image: [main function] check folder of photos or images has the watermark ratio higher or lower than the
                threshold.
    |-crop_coordinate : to crop photo or image from hand-crafted cöordinating sets.
    |-process_region: detect whether the cropped photo or image contain watermark (yellow).

"""

from glob import glob
from typing import Never
import numpy as np
from numpy import uint8
from numpy.typing import NDArray
import cv2

from icecream import ic

def crop_coordinate(image_size: tuple[int, ...]) -> dict[str, list[int, ...] | Never]:

    height, width = image_size
    # (height, width), BL(Bottem-Right) and TL(Top-Left)
    # minx, maxx, miny, maxy
    coordinates = {
        (1152, 2048): {'BR': [1779, 2024, 1058, 1113]},
        (1080, 1920): {'BR': [1587, 1888, 971, 1039], 'TL': [5, 764, 5, 27]}
    }

    # if key not found return empty dict {}
    return coordinates.get((height, width), {})

def process_region(image: NDArray[uint8],
                   coords: list[int, ...],
                   lower: NDArray[uint8],
                   upper: NDArray[uint8]) -> float:
    # BGR2HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # minx, maxx, miny, maxy
    #    0,    1,    2,    3
    region = image[coords[2]:coords[3], coords[0]:coords[1]]
    mask = cv2.inRange(region, lower, upper)
    ratio = cv2.countNonZero(mask) / (region.shape[0] * region.shape[1])
    return ratio

def analyze_image(image_path: str = None,
                  lower: list[int] = None,
                  upper: list[int] = None,
                  threshold: float = 0.3) -> (tuple[int, ...], str | None):

    assert image_path

    # yellow
    if lower is None:
        lower = [22, 93, 0]
    if upper is None:
        upper = [45, 255, 255]

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    image = cv2.imread(image_path)
    (height, width) = image.shape[:2]

    ''' coordinates: {'BR': [1587, 1888, 971, 1039], 'TL': [5, 764, 5, 27]}'''
    coordinates = crop_coordinate((height, width))

    ''''ratios: {'BR': [], 'TL': []}'''
    ratios = {region: [] for region in coordinates}

    for region, coords in coordinates.items():
        ratio = process_region(image, coords, lower, upper)
        ratios[region].append(ratio)

    '''avg_ratios: {'BR': 0.40907758452218096, 'TL': 0.4931129476584022}'''
    avg_ratios = {region: sum(r) / len(r) for region, r in ratios.items() if r}

    ic(avg_ratios)
    if avg_ratios:

        '''mask_region: 'TL'''
        mask_region = max(avg_ratios, key=avg_ratios.get)

        if avg_ratios[mask_region] > threshold:
            # swap position
            return (width, height), mask_region

    return (width, height), None


def analyze_images(folder_path: str = None,
                   ext: str = 'jpg',
                   lower: list[int] = None,
                   upper: list[int] = None,
                   threshold: float = 0.3) -> (tuple[int, ...], str | None):

    assert folder_path

    # yellow
    if lower is None:
        lower = [22, 93, 0]
    if upper is None:
        upper = [45, 255, 255]

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    ratios: dict[str, list[float, ...]] = {}
    image_size: tuple[int, ...] = ()

    for image_path in glob(f'{folder_path}/*.{ext}'):

        image = cv2.imread(image_path)
        (height, width) = image.shape[:2]

        # if image_size is empty, image_size = (height, width) else keeps
        image_size = (height, width) if not image_size else image_size
        assert image_size == (height, width)

        ''' coordinates: {'BR': [1587, 1888, 971, 1039], 'TL': [5, 764, 5, 27]}'''
        coordinates = crop_coordinate((height, width))

        ''''ratios: {'BR': [], 'TL': []}'''
        # if ratios is empty, ratios = {'BR': [], 'TL': []} else keeps
        ratios = {region: [] for region in coordinates} if not ratios else ratios

        for region, coords in coordinates.items():
            ratio = process_region(image, coords, lower, upper)
            ratios[region].append(ratio)

    '''avg_ratios: {'BR': 0.40907758452218096, 'TL': 0.4931129476584022}'''
    avg_ratios = {region: sum(r) / len(r) for region, r in ratios.items() if r}
    # ic(ratios)
    ic(avg_ratios)
    if avg_ratios:
        '''mask_region: 'TL'''
        mask_region = max(avg_ratios, key=avg_ratios.get)

        if avg_ratios[mask_region] > threshold:
            # swap position
            return image_size[::-1], mask_region

    # swap position
    return image_size[::-1], None

# Usage
if __name__ == '__main__':

    folder_path: str = r''

    # image_size, crop_region = analyze_image(image_path)
    image_size, crop_region = analyze_images(folder_path=folder_path, ext='jpg')
