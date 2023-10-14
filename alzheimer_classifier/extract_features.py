import numpy as np
import pandas as pd
import cv2
import zipfile
import os
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class FeatureExtraction:
    def __init__(self, filename=None, extract_to=None):
        """
        Initialize the load_data class.

        Args:
            filename (str): The path to the zip file containing the dataset.
            extract_to (str): The directory where the zip file will be extracted.
        """
        self.filename = filename
        self.extract_to = extract_to

    def unzip_folder(self):
        """
        Load and process the dataset.

        This method unzips the provided zip file, extracts the dataset, and processes it to create training and validation data.

        Raises:
        FolderException: If the dataset folder structure is not as expected.
        """
        self.unzip_file(filename=self.filename, extract_to=self.extract_to)
        # TRAIN_DATA, VAL_DATA = self._extract_features(DIRECTORY = os.path.join(self.extract_to, 'alzheimer_dataset/dataset'))

    def extraction_features(self, DIRECTORY=None, output=''):
        """
                Extract features from the dataset.

                Args:
                    DIRECTORY (str): The path to the directory containing the dataset.

                Raises:
                    FolderException: If the dataset folder structure is not as expected.
        """

        DIRECTORY = DIRECTORY
        CATEGORIES = ['mild', 'moderate', 'no', 'very_mild']
        VAL_DATA = []
        TRAIN_DATA = []

        for sub_folder in os.listdir(DIRECTORY):
            if sub_folder == 'test':
                print("\t" * 5, "{} folder is accessing".format(sub_folder).upper())
                print("\t" * 2, "_" * 80, '\n')
                FULL_PATH = os.path.join(DIRECTORY, sub_folder)
                for category in CATEGORIES:
                    IMAGE_PATH = os.path.join(FULL_PATH, category)
                    for image_filename in os.listdir(IMAGE_PATH):
                        IMAGE_PATH = os.path.join(
                            FULL_PATH, category, image_filename)
                        if os.path.exists(IMAGE_PATH):
                            image_array = cv2.resize(cv2.imread(IMAGE_PATH),
                                                     dsize=(120, 120))
                            image_label = CATEGORIES.index(category)
                            VAL_DATA.append([image_array,
                                             image_label])
                        else:
                            print(
                                f"Warning: Image file not found at path {IMAGE_PATH}")

                    print("\t" * 5, "{} folder is completed.\n".format(category))

                print("\t" * 2, "_" * 80, '\n')

            if sub_folder == 'train':
                print("\t"*5, "{} folder is accessing".format(sub_folder).upper())
                print("\t" * 2, "_" * 80, '\n')
                FULL_PATH = os.path.join(DIRECTORY, sub_folder, output)
                for category in CATEGORIES:
                    IMAGE_PATH = os.path.join(FULL_PATH, category)
                    for image_filename in os.listdir(IMAGE_PATH):
                        IMAGE_PATH = os.path.join(
                            FULL_PATH, category, image_filename)
                        if os.path.exists(IMAGE_PATH):
                            image_array = cv2.resize(cv2.imread(IMAGE_PATH),
                                                     dsize=(120, 120))
                            image_label = CATEGORIES.index(category)
                            TRAIN_DATA.append([image_array,
                                               image_label])
                        else:
                            print(
                                f"Warning: Image file not found at path {IMAGE_PATH}")

                    print("\t" * 5, "{} folder is completed.\n".format(category))

        print("\t" * 2, "_" * 80, '\n')
        print("\t" * 5, "Details of dataset".upper())
        print("\t" * 2, "_" * 80, '\n')

        print("\t" * 5, "Length of train data # {} ".format(len(TRAIN_DATA)), '\n')
        print("\t" * 5, "Length of validation data # {} ".format(len(VAL_DATA)), '\n')

        return TRAIN_DATA, VAL_DATA

    def _check_augmentationed(self, DIRECTORY=None):
        # E:/alzheimer_classifier/alzheimer_dataset/dataset
        # E:/alzheimer_classifier/alzheimer_dataset/dataset/train
        # train/output
        # test
        # UPDATE_DIRECTORY = os.path.join(DIRECTORY, 'train/output')
        CONFIRM = 0

        for folder in os.listdir(DIRECTORY):
            FOLDER = os.path.join(DIRECTORY, folder)
            if os.path.isdir(FOLDER):
                CONFIRM = CONFIRM + 1

        if CONFIRM == 2:
            train_data, val_data = self.extraction_features(
                DIRECTORY=DIRECTORY, output='output')

            return train_data, val_data
        else:
            raise Exception('''Folder Format like\n:
                                    xyz.zip
                                        |
                                    dataset(This folder name should be kept)
                                        |__train
                                        |   |__very_mild(here all the images would be kept)
                                        |   |__no(here all the images would be kept)
                                        |   |__mild(here all the images would be kept)
                                        |   |__moderate(here all the images would be kept)
                                        |__test
                                            |__very_mild(here all the images would be kept)
                                            |__no(here all the images would be kept)
                                            |__mild(here all the images would be kept)
                                            |__moderate(here all the images would be kept)
                                    ''')

    def unzip_file(self, filename=None, extract_to=None):
        """
            Extracts the zip file to the specified directory.

            Args:
                filename (str): The path to the zip file.
                extract_to (str): The directory where the zip file will be extracted.
        """
        link_folder = filename
        if link_folder.split(".")[-1] == 'zip':
            print(
                "Unzipping is in progress. It will take some time, so please be patient.\n")
            with zipfile.ZipFile(link_folder, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            raise Exception("File should be in zip format")


if __name__ == "__main__":
    FeatureExtraction(filename='E:/alzheimer_classifier/alzheimer_dataset.zip',
                      extract_to='E:/alzheimer_classifier/')
