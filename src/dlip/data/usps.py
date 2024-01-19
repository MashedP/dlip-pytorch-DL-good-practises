# import torchvision
# import torchvision.transforms as transforms

import logging
import os
from pathlib import Path

import requests


def download_usps(path):
    """Download the USPS Dataset
    * Handwritten digits with 10 classes
    * 16x16 pixels for each image
    * 6 000 data examples in training set, 1 291 examples in validation set, 2 007 in test set

    Args:
        path (String or Path): Path to save the dataset
    """
    current_path = Path(path)
    writing_path = current_path.joinpath("USPS/")
    if not os.path.exists(writing_path.joinpath("usps.bz2")):
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2"
        r = requests.get(url, allow_redirects=True)
        if not os.path.isdir(writing_path):
            logging.info("Creating Folder..")
            os.mkdir(writing_path)
        open(writing_path.joinpath("usps.bz2"), "wb").write(r.content)
        logging.info(f"File downloaded in {writing_path}")
    else:
        logging.info(f"Found already existing USPS dataset at {writing_path}")
