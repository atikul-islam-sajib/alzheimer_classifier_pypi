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
from alzheimer_classifier.extract_features import FeatureExtraction
import Augmentor

class Augmentation:
    def __init__(self, filename = None, extract_to = None):
        """
        Initialize the load_data class.

        Args:
            filename (str): The path to the zip file containing the dataset.
            extract_to (str): The directory where the zip file will be extracted.
        """
        self.filename   = filename
        self.extract_to = extract_to
    
    def perform_augmentation(self, samples = 1000):
        """
        Apply image augmentation to a dataset for training or preprocessing.

        This method applies various image augmentation techniques to a dataset located in
        the specified directory. The augmented images are saved in the same directory.

        Parameters:
            - samples (int): The number of augmented images to generate and save.

        Example usage:
            feature_extraction = FeatureExtraction(filename=self.filename, extract_to=self.extract_to)
            feature_extraction.unzip_folder()

            # Define the directory containing the dataset for augmentation
            DIRECTORY = os.path.join(self.extract_to, 'alzheimer_dataset/dataset/train')

            # Create an Augmentor pipeline with various augmentation operations
            pipeline = Augmentor.Pipeline(DIRECTORY)

            # Define augmentation operations, adjust probabilities and parameters as needed
            pipeline.rotate(probability=0.3, max_left_rotation=10, max_right_rotation=10)
            pipeline.flip_left_right(probability=0.3)
            pipeline.crop_random(probability=0.1, percentage_area=0.7)
            pipeline.resize(probability=0.1, width=256, height=256)
            pipeline.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)
            pipeline.random_color(probability=0.5, min_factor=0.8, max_factor=1.2)
            pipeline.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)
            pipeline.zoom(probability=0.7, min_factor=0.9, max_factor=1.1)

            # Generate and save the augmented images
            pipeline.sample(samples)  # Replace 1000 with the desired number of augmented images

        Note:
        - The 'DIRECTORY' variable should be set to the directory containing the dataset
        you want to augment.
        - Adjust the probability and parameters of each augmentation operation according to
        your specific requirements.
        - The 'samples' parameter controls how many augmented images will be generated and
        saved in the 'DIRECTORY'.
        """
        feature_extraction = FeatureExtraction(filename = self.filename, extract_to = self.extract_to)
        feature_extraction.unzip_folder()
        
        DIRECTORY = os.path.join(self.extract_to, 'alzheimer_dataset/dataset/train')
        pipeline = Augmentor.Pipeline(DIRECTORY)

        # Rotation
        pipeline.rotate(probability=0.3, max_left_rotation=10, max_right_rotation=10)

        # Flip Left-Right
        pipeline.flip_left_right(probability=0.3)

        # Random Cropping with increased percentage_area
        pipeline.crop_random(probability=0.1, percentage_area=0.7)

        # Resize to a specific size (adjust to match your model's input size)
        pipeline.resize(probability=0.1, width=256, height=256)

        # Random Brightness, Color, and Contrast
        pipeline.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)
        pipeline.random_color(probability=0.5, min_factor=0.8, max_factor=1.2)
        pipeline.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)

        # Zoom within a reasonable range
        pipeline.zoom(probability=0.7, min_factor=0.9, max_factor=1.1)

        # Sample code for applying the augmentation
        pipeline.sample(samples)  # Replace 1000 with the desired number of augmented images
        

if __name__ == "__main__":
    augmentation = Augmentation(filename   = 'E:/alzheimer_classifier/alzheimer_dataset.zip',
                                extract_to = 'E:/alzheimer_classifier/')
    augmentation.perform_augmentation(samples = 10 * 2)