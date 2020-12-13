# This file is for DATA PREPROCESSING and DATA ANALYSIS (AUC Score)

import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import PIL
import seaborn
import scipy.io as sio
from numpy import ndarray
from PIL import Image, ImageFile, ImageOps
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


IMGPATH: str = str(Path(__file__).resolve().parent.parent / "data/")
SHROOMPATH: str = str(Path(__file__).resolve().parent.parent / "data/all_shrooms.mat")

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allowing for truncated files to be processed
CNN = False

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
            img: Any = Image.open(image)
            img = img.resize((300, 300), PIL.Image.ANTIALIAS)
            img.save(IMGPATH + "/%s/%s%s%s" % (species, species, str(imgcount), ".jpg"))
            os.remove(image)
            imgcount += 1


def img_to_mat() -> None:
    """Reading in jpg images and converting to a simple mat file holding
    numpy arrays like {0: (instances, 300, 300) where the key int represents a class (species)"""

    species_dict = {}

    for species in SHROOMS.keys():
        np_data: List = []
        image_paths: Generator = Path(IMGPATH).rglob("%s/*.jpg" % species)
        count = 0
        # Iterating through each subdirectory of Mushroom species
        for image in image_paths:
            if count == 11:
                break
            # Appending each numpy representation of image data to list
            image: Any = Image.open(image)
            # Grayscaling bc I don't want to deal with RGB shape shenanigans
            # image: Any = PIL.ImageOps.grayscale(image)
            img: np.ndarray = np.array(image)
            np_data.append(img)  # List of all npy array images
            count += 1

        # Dictionary where {'0': [array of (instances, 300, 300)] }
        img_npy: np.ndarray = np.array(np_data)
        species_dict[str(SHROOMS[species])] = img_npy
    # Create a mat file like mnist dataset
    sio.savemat(IMGPATH + "seg_shrooms.mat", species_dict)


def load_shroom_data() -> Tuple[np.ndarray, np.ndarray]:
    # Loading .mat data and extracting X/y data to a numpy array
    shroom_dic: Dict[str, ndarray] = sio.loadmat(SHROOMPATH)

    # reshaping data to (instances-inorder, 9000) and (classes-inorder,)
    X: List[np.ndarray] = []
    y: List[np.ndarray] = []
    for i in range(9):

        instances = shroom_dic[str(i)].shape[0]

        if CNN:
            # reshaping data to (instances-inorder, 300, 300) and (classes-inorder,)
            temp = shroom_dic[str(i)]
            temp = np.array(shroom_dic[str(i)])[:, :, :, np.newaxis]
            X.append(temp)

        if not CNN:
            # X = (instances, 9000)
            X.append(
                shroom_dic[str(i)].reshape(
                    [
                        instances,
                        np.prod(shroom_dic[str(i)].shape[1:]),
                    ]
                )
            )

        # y = (instances,)
        y.append(np.full(shape=instances, fill_value=i))

    # Y.shape = (5816,)
    y: np.ndarray = np.concatenate(y)
    # CNN: X.shape = (5816, 300, 300, 1)
    # not CNN: X.shape = (5816, 9000)
    X: np.ndarray = np.concatenate(X)

    return X, y


def shroom_auc() -> None:
    """ This function produced a heat map of AUC values for each pixel in the 300x300 images.  """
    # features = 9000 pixels
    # data = every instance of it
    img_arr, labels = load_shroom_data()  # y = (5816,)
    img_arr = img_arr.T  # (5816, 9000) -> (9000, 5816)
    auc_val: List = []
    for pixel in img_arr:
        # This line is from Derek's tutorial
        auc: float = mannwhitneyu(pixel, labels).statistic / (len(pixel) * len(labels))
        auc_val.append(auc - 0.50)
        # Sorting AUC values by furthest proximity to 0.50
    print(set(auc_val))
    auc_img: np.ndarray = np.array(auc_val)
    auc_img = Image.fromarray(auc_img.reshape(300, 300))
    plt.imsave("aucplot.png", auc_img, cmap="seismic")
