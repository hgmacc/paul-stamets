import glob
import os
import sys
from pathlib import Path
from subprocess import check_output
from time import sleep, time
from typing import Any, Dict, Generator, List, Tuple
import scipy
import scipy.io as sio

import matplotlib as plt
import numpy as np
import PIL
from IPython.display import Image as _Imgdis
from IPython.display import display
from numpy import ndarray
from PIL import Image, ImageOps, ImageFile
from skimage import color, data
from skimage.transform import rescale, resize

IMGPATH: str = str(Path(__file__).resolve().parent / "data/Mushrooms/")
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allowing for truncated files to be processed
CNN = True
SHROOMS: Dict[str, int] = {
    "Agaricus": 0,
    "Amanita": 1,
    "Boletus": 2,
    "Cortinarius": 3,
    "Entoloma": 4,
    "Hygrocybe": 5,
    "Lactarius": 6,
    "Russula": 7,
    "Suillus": 8,
}


def img_resize() -> None:
    """
    Used to convert original images with varied dimensions to uniform shape (300x300).
    Also renaming files (files we're coming up with I/O issues when converting to .npy)
    """
    for species in SHROOMS:
        image_paths: Generator = Path(IMGPATH).rglob("%s/*.jpg" % species)
        imgcount = 0
        # Sifting through all images
        for image in image_paths:
            print(image)
            img: Any = Image.open(image)
            img = img.resize((300, 300), PIL.Image.ANTIALIAS)
            img.save(IMGPATH + "/%s/%s%s%s" % (species, species, str(imgcount), ".jpg"))
            os.remove(image)
            imgcount += 1


def img_to_npy() -> None:
    """Reading in jpg images and converting to a simple mat file holding
    numpy arrays like {0: (instances, 300, 300) where the key int represents a class (species)"""

    species_dict = {}

    for species in SHROOMS.keys():
        np_data: List = []
        image_paths: Generator = Path(IMGPATH).rglob("%s/*.jpg" % species)

        # Iterating through each subdirectory of Mushroom species
        for image in image_paths:
            # Appending each numpy representation of image data to list
            image: Any = Image.open(image)
            # Grayscaling bc I don't want to deal with RGB shape shenanigans
            image: Any = PIL.ImageOps.grayscale(image)
            img: np.ndarray = np.array(image)
            np_data.append(img)  # List of all npy array images

        # Dictionary where {'0': [array of (instances, 300, 300)] }
        img_npy: np.ndarray = np.array(np_data)
        species_dict[str(SHROOMS[species])] = img_npy

    for i in species_dict.keys():
        print(i)
        print(species_dict[i].shape)
    # Create a mat file like mnist dataset
    sio.savemat(IMGPATH + "shrooms.mat", species_dict)


# Loading .mat data and extracting X/y data to a numpy array
num_dic: Dict[str, ndarray] = sio.loadmat(NUMFILEPATH)

if q == "all":
    # reshaping data to (30000, 784) and (30000,)
    num_data: ndarray = (
        num_dic["X"]
        .reshape([np.prod(num_dic["X"].shape[:2]), num_dic["X"].shape[-1]])
        .T
    )
    num_labels: ndarray = num_dic["y"][0]

elif q == "CNN":
    # reshaping data to (30000, 784) and (30000,)
    num_data = num_dic["X"].transpose([2, 0, 1]).astype("float32")
    # Adding acis for CNN
    num_data = np.array(num_data)[:, :, :, np.newaxis]
    num_labels = num_dic["y"].flatten()


# Find which species are poisonous
# Try pre-trained model

# Try pre-trained model, mix mushroom imgs with non mush and see if it works
