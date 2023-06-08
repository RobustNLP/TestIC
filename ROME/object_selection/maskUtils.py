import pycocotools.mask as mask_util
import numpy as np
from typing import List

# These functions are extracted from Detectron2, which serve as basic functions to convert 
# polygon or rle format into bitmasks
# Please refer to https://github.com/facebookresearch/detectron2 for more information 

def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.uint8)

def rle_to_bitmask(rle):
    rle = mask_util.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])

    return mask_util.decode(rle).astype(np.uint8)
