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
from alzheimer_classifier.augmentation import Augmentation
from alzheimer_classifier.extract_features import FeatureExtraction


class EmptyListException(Exception):
    def __init__(self, message="A custom exception"):
        super().__init__(message)


class load_data:
    def __init__(self, filename=None, extract_to=None):
        """
            Initialize the load_data class.

            Args:
                filename (str): The path to the zip file containing the dataset.
                extract_to (str): The directory where the zip file will be extracted.
        """
        self.filename = filename
        self.extract_to = extract_to
        self.batch_size = 128

        self.X_train = list()
        self.y_train = list()
        self.X_val = list()
        self.y_val = list()

    def data_augmentation(self, samples = 1000):
        """
        Perform data augmentation on a specified file.

        This method initializes an Augmentation object and performs data augmentation
        on the specified file by creating augmented samples. The augmented samples are
        written to a designated location.

        Parameters:
            - filename (str): The path to the input data file to be augmented.
            - extract_to (str): The directory where augmented samples will be saved.

        Example usage:
            augmentation = Augmentation(filename=self.filename, extract_to=self.extract_to)
            augmentation.perform_augmentation(samples=10 * 2)

        The `perform_augmentation` method of the Augmentation object is responsible for
        generating augmented samples, and the number of samples to create is determined
        by the `samples` parameter.

        Note:
        - Make sure that the 'filename' and 'extract_to' parameters are properly set
        before calling this method to ensure that the augmentation is performed on
        the correct data and the results are saved in the desired location.
        - The 'samples' parameter specifies the number of augmented samples to create.
        """
        augmentation = Augmentation(filename=self.filename,
                                    extract_to=self.extract_to)
        augmentation.perform_augmentation(samples=samples)

    def dataloader(self, batch_size = 128):
        """
            Load and process the dataset.

            This method performs the following steps:
            1. Unzips the provided zip file to the specified extraction directory.
            2. Checks if data augmentation output folders exist in the dataset. If found,
            it loads data from these folders. If not found, it extracts features
            from the dataset images.
            3. Preprocesses the training and validation data to create data loaders.

            Returns:
                Tuple (TRAIN_LOADER, TEST_LOADER): Training and validation data loaders.

            Raises:
                FolderException: If the dataset folder structure is not as expected.
        """
        # Define the batch_size of the dataset
        self.batch_size = batch_size
        feature_extraction = FeatureExtraction(
            filename=self.filename, extract_to=self.extract_to)
        feature_extraction.unzip_folder()

        if 'output' in os.listdir(os.path.join(self.extract_to, 'alzheimer_dataset/dataset/', 'train')):
            train_data, val_data = feature_extraction._check_augmentationed(
                DIRECTORY=os.path.join(self.extract_to, 'alzheimer_dataset/dataset'))
            TRAIN_LOADER, TEST_LOADER = self._preprocessing_dataset(
                train_data=train_data,
                val_data=val_data,
                batch_size = self.batch_size)

            return TRAIN_LOADER, TEST_LOADER
        else:
            train_data, val_data = feature_extraction.extraction_features(
                DIRECTORY=os.path.join(self.extract_to, 'alzheimer_dataset/dataset'
                                       ))
            TRAIN_LOADER, TEST_LOADER = self._preprocessing_dataset(
                train_data=train_data,
                val_data=val_data,
                batch_size = self.batch_size)

            return TRAIN_LOADER, TEST_LOADER

    def _preprocessing_dataset(self, train_data=None, val_data=None, batch_size = 128):

        # Randomly shuffle the data
        random.shuffle(train_data)
        random.shuffle(val_data)

        # Normalize the data (assuming you have a method for this)
        X_train, y_train, X_val, y_val = self._split_independent_dependent(TRAIN_DATA=train_data,
                                                                           VAL_DATA=val_data)
        X_train, X_val, y_train, y_val = self._do_normalization(X_train=X_train,
                                                                X_val=X_val,
                                                                y_train=y_train,
                                                                y_val=y_val)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        TRAIN_LOADER, TEST_LOADER, VAL_LOADER = self._create_dataloader(
            X=X_train, y=y_train, X_val=X_val, y_val=y_val, batch_size = batch_size)

        return TRAIN_LOADER, TEST_LOADER

    def _split_independent_dependent(self, TRAIN_DATA=None, VAL_DATA=None):
        """
            Split the dataset into independent (features) and dependent (labels) sets.

            Args:
                TRAIN_DATA (list): A list of training data samples, where each element is a tuple of
                    an independent feature and its corresponding dependent label.
                VAL_DATA (list): A list of validation data samples, where each element is a tuple of
                    an independent feature and its corresponding dependent label.

            Returns:
                tuple: A tuple containing the following elements:
                    - X_train (list): A list of independent features for training.
                    - y_train (list): A list of dependent labels for training.
                    - X_val (list): A list of independent features for validation.
                    - y_val (list): A list of dependent labels for validation.

            Raises:
                EmptyListException: If either TRAIN_DATA or VAL_DATA is empty.
        """
        X_train = []
        y_train = []

        X_val = []
        y_val = []

        if TRAIN_DATA:
            for (independent, dependent) in TRAIN_DATA:
                X_train.append(independent)
                y_train.append(dependent)
        else:
            raise EmptyListException("Empty list".capitalize())

        if VAL_DATA:
            for (independent, dependent) in VAL_DATA:
                X_val.append(independent)
                y_val.append(dependent)
        else:
            raise EmptyListException("Empty list".capitalize())

        train = len(np.unique(np.array(y_train)))
        test = len(np.unique(np.array(y_val)))

        assert train == 4 and test == 4
        print("\t" * 5, "Total number of target class # {} ".format(4), '\n')

        return X_train, y_train, X_val, y_val

    def _do_normalization(self, X_train=None, X_val=None, y_train=None, y_val=None):
        """
            Normalize the pixel values of the dataset.

            This method scales the pixel values of the independent features in the dataset to the
            range [0, 1] by dividing them by 255.

            Args:
                X_train (list or np.ndarray): A list or numpy array of independent features for training.
                y_train: Unused in this method.
                X_val (list or np.ndarray): A list or numpy array of independent features for validation.
                y_val: Unused in this method.

            Returns:
                None

            Notes:
                - y_train and y_val are not used in this method, and the normalization is applied only
                to X_train and X_val.
        """
        X_train = np.array(X_train)
        X_val = np.array(X_val)

        y_train = np.array(y_train)
        y_val = np.array(y_val)

        X_train = X_train/255
        X_val = X_val/255

        return X_train, X_val, y_train, y_val

    def show_plot(self):
        """
            Display a grid of sample images and their corresponding labels.

            This method shows a grid of sample images from the training data and their corresponding labels.
            It can be useful for visually inspecting a subset of the dataset.

            Returns:
                None
        """
        sample_data = self.X_train[0:20]
        sample_label = self.y_train[0:20]

        plt.figure(figsize=(12, 8))

        for index, image in enumerate(sample_data):
            plt.subplot(4, 5, index + 1)
            plt.imshow(image)
            plt.title('Mild' if sample_label[index] == 0
                      else 'Moderate' if sample_label[index] == 1
                      else 'No' if sample_label[index] == 2
                      else 'Very Mild'
                      )
            plt.tight_layout()
            plt.axis("off")

        plt.show()

    def show_distribution(self):
        """
            Visualize the distribution of labels in the training and validation datasets.

            This method creates bar plots to show the distribution of labels in the training and validation datasets.
            It provides insights into the class distribution, which can be helpful for understanding dataset balance.

            Returns:
                None
        """
        try:

            df = pd.DataFrame(self.y_train, columns=['train_target'])
            plt.title("Distribution of Train dataset.\n".capitalize())

            df.loc[:, 'train_target'].map({0: 'Mild', 1: 'Moderate', 2: 'No', 3: 'Very Mild'}).\
                value_counts().plot(kind='barh')

            plt.xlabel("Distribution of train label")
            plt.show()

            print("\n\n")

            try:
                df = pd.DataFrame(self.y_val, columns=['val_target'])
                plt.title("Distribution of validation dataset.\n".capitalize())

                df.loc[:, 'val_target'].map({0: 'Mild', 1: 'Moderate', 2: 'No', 3: 'Very Mild'}).\
                    value_counts().plot(kind='barh')

                plt.xlabel("Distribution of validation label")
                plt.show()
            except EmptyListException as e:
                print("A custom exception # {}".format(e))
            except Exception as e:
                print(f"An error occurred: {str(e)}")

        except EmptyListException as e:
            print("A custom exception # {}".format(e))
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def _create_dataloader(self, X=None, y=None, X_val=None, y_val=None, batch_size = 128):
        """
            Data Preprocessing and Preparation for Deep Learning Model

            This code performs the following tasks:

            1. Data Preprocessing:
            - Reshapes and converts the training and validation data into PyTorch tensors.

            2. Data Splitting:
            - Splits the data into training and testing sets using train_test_split.

            3. Data Loading:
            - Creates data loaders for training, testing, and validation datasets using DataLoader.

            4. Information Display:
            - Displays dataset shapes and data loader batch sizes.

            5. Data Extraction:
            - Extracts data and labels from the data loaders.

            6. Returns:
            - Returns the created data loaders for further model training and evaluation.
        """
        CHANNEL = 3
        HEIGHT  = 120
        WIDTH   = 120
        BATCH_SIZE = batch_size

        X = X.reshape(X.shape[0], CHANNEL, HEIGHT, WIDTH)
        X = torch.tensor(data=X, dtype=torch.float32)

        X_val = X_val.reshape(X_val.shape[0], CHANNEL, HEIGHT, WIDTH)
        X_val = torch.tensor(data=X_val, dtype=torch.float32)

        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X,
                                                                                y,
                                                                                test_size=0.30,
                                                                                random_state=42)

        print("\t" * 5, "X_train (TRAIN DATASET) shape  # {} ".format(X_train_data.shape), '\n')
        print("\t" * 5, "y_train (TRAIN DATASET) shape  # {} ".format(y_train_data.shape), '\n')
        print(
            "\t" * 5, "X_test  (TRAIN DATASET) shape  # {} ".format(X_test_data.shape), '\n')
        print(
            "\t" * 5, "y_test  (TRAIN DATASET) shape  # {} ".format(y_test_data.shape), '\n')

        TRAIN_LOADER = DataLoader(dataset=list(zip(X_train_data, y_train_data)),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

        TEST_LOADER = DataLoader(dataset=list(zip(X_test_data, y_test_data)),
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

        VAL_LOADER = DataLoader(dataset=list(zip(X_val, y_val)),
                                batch_size=BATCH_SIZE,
                                shuffle=True)

        print("\t" * 2, "_" * 80, '\n')
        print("\t" * 5, "Batch size of Train # {} ".format(TRAIN_LOADER.batch_size), '\n')
        print("\t" * 5, "Batch size of Test  # {} ".format(TEST_LOADER.batch_size), '\n')

        print("\t" * 2, "_" * 80, '\n')

        # Extract the data and label
        train_data, train_label = next(iter(TRAIN_LOADER))
        test_data, test_label = next(iter(TEST_LOADER))
        val_data, val_label = next(iter(VAL_LOADER))

        print("\t" * 5, "Train data  (TRAIN DATASET) with single batch_size  # {} ".format(train_data.shape), '\n')
        print("\t" * 5, "Train label (TRAIN DATASET) with single batch_size  # {} ".format(train_label.shape), '\n')
        print("\t" * 5, "Test data   (TRAIN DATASET) with single batch_size  # {} ".format(test_data.shape), '\n')
        print("\t" * 5, "Test label  (TRAIN DATASET) with single batch_size  # {} ".format(test_label.shape), '\n')

        print("\t" * 2, "_" * 80, '\n')

        print("\t" * 5, "Val data  (VALIDATION DATASET) with single batch_size  # {} ".format(val_data.shape), '\n')
        print("\t" * 5, "VAL label (VALIDATION DATASET) with single batch_size  # {} ".format(val_label.shape), '\n')

        return TRAIN_LOADER, TEST_LOADER, VAL_LOADER


if __name__ == "__main__":
    loader = load_data(filename='D:/alzheimer_dataset/alzheimer_dataset.zip',
                       extract_to='D:/alzheimer_dataset/')

    loader.data_augmentation(samples = 10240*2)
    # train_loader, test_loader = loader.dataloader()
    loader.dataloader(batch_size = 256)

    loader.show_plot()
    loader.show_distribution()
