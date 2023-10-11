import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.optim as optim


class Trainer:
    """
      A class for training multiple models with shared parameters.

      Args:
          classifier (nn.Module, optional): The shared classifier model. Default is None.

      Attributes:
          model1_loss_function (nn.Module): Loss function for model 1.
          model2_loss_function (nn.Module): Loss function for model 2.
          model3_loss_function (nn.Module): Loss function for model 3.
          model1_lr (float): Learning rate for model 1.
          model2_lr (float): Learning rate for model 2.
          model3_lr (float): Learning rate for model 3.
          model1_optimizer (optim.Optimizer): Optimizer for model 1.
          model2_optimizer (optim.Optimizer): Optimizer for model 2.
          model3_optimizer (optim.Optimizer): Optimizer for model 3.
          MODEL1_ACCURACY (list): List to store accuracy values for model 1.
          MODEL2_ACCURACY (list): List to store accuracy values for model 2.
          MODEL3_ACCURACY (list): List to store accuracy values for model 3.
          MODEL1_TOTAL_LOSS (list): List to store total loss values for model 1.
          MODEL2_TOTAL_LOSS (list): List to store total loss values for model 2.
          MODEL3_TOTAL_LOSS (list): List to store total loss values for model 3.

      Raises:
          ValueError: If the `classifier` is not provided, an exception is raised with an error message.

    """

    def __init__(self, classifier=None, device=None):
        self.classifier = classifier
        self.device = device

        self.model1_loss_function = nn.CrossEntropyLoss()
        self.model2_loss_function = nn.CrossEntropyLoss()
        self.model3_loss_function = nn.CrossEntropyLoss()

        if classifier:
            self.model1_lr = 0.001
            self.model2_lr = 0.001
            self.model3_lr = 0.001

            self.model1_optimizer = optim.Adam(
                params=self.classifier.parameters(), lr=self.model1_lr)
            self.model2_optimizer = optim.Adam(
                params=self.classifier.parameters(), lr=self.model2_lr)
            self.model3_optimizer = optim.Adam(
                params=self.classifier.parameters(), lr=self.model3_lr)
        else:
            raise "model is not defined".title()

        self.history = {'m1_train_loss': [],
                        'm2_train_loss': [],
                        'm3_train_loss': [],
                        'm1_train_acc' : [],
                        'm2_train_acc' : [],
                        'm3_train_acc' : [],
                        'm1_val_loss'  : [],
                        'm2_val_loss'  : [],
                        'm3_val_loss'  : [],
                        'm1_val_acc'   : [],
                        'm2_val_acc'   : [],
                        'm3_val_acc'   : []
                        }
        
        self.get_models = list()

    def _connect_GPU(self, independent_data=None, dependent_data=None):
        """
        Connects the provided classifier and data to a GPU device, and ensures dependent_data is of 'long' type.

        :param classifier: The classifier to be moved to the GPU device.
        :param independent_data: The independent data to be moved to the GPU device.
        :param dependent_data: The dependent data to be converted to 'long' type and moved to the GPU device.

        :return: A tuple containing the independent data and dependent data, both residing on the GPU device.
        """
        independent_data = independent_data.to(self.device)
        dependent_data = dependent_data.long()
        dependent_data = dependent_data.to(self.device)

        return independent_data, dependent_data

    def _l1_regularization(self, model=None, lambda_value=0.01):
        """
        Compute L1 regularization for the model's parameters.

        :param model: The model for which L1 regularization is computed.
        :param lambda_value: The regularization strength (lambda value).

        :return: The L1 regularization term as a scalar.
        """
        return sum(torch.norm(parameter, 1) for parameter in model.parameters())

    def _l2_regularization(self, model=None, lambda_value=0.01):
        """
        Compute L2 regularization for the model's parameters.

        :param model: The model for which L1 regularization is computed.
        :param lambda_value: The regularization strength (lambda value).

        :return: The L2 regularization term as a scalar.
        """
        return sum(torch.norm(parameter, 2) for parameter in model.parameters())

    def _do_back_propagation(self, optimizer=None, model_loss=None):
        """
        Perform backpropagation to update model parameters.

        :param optimizer: The optimizer used for updating model parameters.
        :param model_loss: The loss computed for the model.

        This function performs the following steps:
        1. Zeroes out the gradients in the optimizer.
        2. Backpropagates the model_loss to compute gradients.
        3. Updates the model parameters using the optimizer.

        """
        optimizer.zero_grad()
        model_loss.backward(retain_graph=True)
        optimizer.step()

    def _compute_predicted_label(self, model_prediction=None):
        """
        Compute predicted labels from the model's predictions.

        :param model_prediction: The model's output predictions.

        :return: The computed predicted labels as a NumPy array.
        """
        model_predicted = torch.argmax(model_prediction, dim=1)
        model_predicted = model_predicted.cpu().detach().flatten().numpy()

        return model_predicted

    def _compute_actual_label(self, actual_label=None):
        """
        Extract the actual labels from a tensor and convert them to a NumPy array.

        :param actual_label: The tensor containing the actual labels.

        :return: The actual labels as a NumPy array.
        """
        return actual_label.cpu().detach().flatten().numpy()

    def _compute_model_loss(self, model=None, loss_function=None, actual_label=None):
        """
        Computes the loss of a model given the actual labels using a specified loss function.

        :param model: The model for which the loss is to be computed.
        :param loss_function: The loss function used to compute the loss.
        :param actual_label: The actual labels for comparison.

        :return: The computed loss value.
        """
        return loss_function(model, actual_label)

    def _model_accuracy(self, actual_label=None, predicted_label=None):
        """
        Compute the accuracy of a model's predictions by comparing them to the actual labels.

        :param actual_label: The actual labels.
        :param predicted_label: The predicted labels.

        :return: The accuracy score as a float.
        """
        return accuracy_score(actual_label, predicted_label)

    def _display(self, model1_train_loss = None,
                 model2_train_loss       = None,
                 model3_train_loss       = None,
                 model1_train_acc        = None,
                 model2_train_acc        = None,
                 model3_train_acc        = None,
                 model1_val_loss         = None,
                 model2_val_loss         = None,
                 model3_val_loss         = None,
                 model1_val_acc          = None,
                 model2_val_acc          = None,
                 model3_val_acc          = None,
                 running_epochs          = None,
                 total_epochs            = None):
        """
        Display training and validation metrics for multiple models during the training process.

        :param model1_train_loss: Training loss for model 1.
        :param model2_train_loss: Training loss for model 2.
        :param model3_train_loss: Training loss for model 3.
        :param model1_train_acc: Training accuracy for model 1.
        :param model2_train_acc: Training accuracy for model 2.
        :param model3_train_acc: Training accuracy for model 3.
        :param model1_val_loss: Validation loss for model 1.
        :param model2_val_loss: Validation loss for model 2.
        :param model3_val_loss: Validation loss for model 3.
        :param model1_val_acc: Validation accuracy for model 1.
        :param model2_val_acc: Validation accuracy for model 2.
        :param model3_val_acc: Validation accuracy for model 3.
        :param running_epochs: Current epoch number.
        :param total_epochs: Total number of epochs.

        This function displays training and validation metrics for multiple models in a specific format.
        """

        print("Epochs: {}/{} ".format(running_epochs + 1, total_epochs))

        print("[================] m1_loss: {:.4f} - m1_acc: {:.4f} - "
              "m2_loss: {:.4f} - m2_acc: {:.4f} - "
              "m3_loss: {:.4f} - m3_acc: {:.4f} - "
              "val1_loss: {:.4f} - val1_acc: {:.4f} - "
              "val2_loss: {:.4f} - val2_acc: {:.4f} - "
              "val3_loss: {:.4f} - val3_acc: {:.4f}".format(
                  np.array(model1_train_loss).mean(),
                  model1_train_acc,
                  np.array(model2_train_loss).mean(),
                  model2_train_acc,
                  np.array(model3_train_loss).mean(),
                  model3_train_acc,
                  np.array(model1_val_loss).mean(),
                  model1_val_acc,
                  np.array(model2_val_loss).mean(),
                  model2_val_acc,
                  np.array(model3_val_loss).mean(),
                  model3_val_acc
              ))

    def performance(self):
        """
        Plot training and validation loss and accuracy for three different models.
        """
        # Create a subplot with 2 rows and 3 columns
        fig, axis = plt.subplots(2, 3, figsize=(16, 6))

        # Define model names and colors
        model_names = ['Model 1', 'Model 2', 'Model 3']
        colors = ['b', 'g', 'r']

        for i in range(3):
            # Plot training and validation loss for each model
            axis[0][i].plot(self.history[f'm{i+1}_train_loss'],
                            label=f'{model_names[i]} Train Loss', color=colors[i])
            axis[0][i].plot(
                self.history[f'm{i+1}_val_loss'], label=f'{model_names[i]} Validation Loss', linestyle='--', color=colors[i])
            axis[0][i].set_title(f'{model_names[i]} Loss')
            axis[0][i].legend()

            # Plot training and validation accuracy for each model
            axis[1][i].plot(
                self.history[f'm{i+1}_train_acc'], label=f'{model_names[i]} Train Accuracy', color=colors[i])
            axis[1][i].plot(
                self.history[f'm{i+1}_val_acc'], label=f'{model_names[i]} Validation Accuracy', linestyle='+', color=colors[i])
            axis[1][i].set_title(f'{model_names[i]} Accuracy')
            axis[1][i].legend()

        plt.tight_layout()
        plt.show()

    def train(self, train_loader=None, test_loader=None, epochs=None):
        """
        Train multiple models and evaluate their performance over a specified number of epochs.

        Args:
            TRAIN_LOADER: DataLoader for the training dataset.
            TEST_LOADER: DataLoader for the testing dataset.
            TOTAL_EPOCHS: Total number of epochs for training.

        This method iteratively trains and evaluates multiple models over a specified number of epochs.
        It stores and updates training and validation metrics in a history dictionary.
        """

        TRAIN_LOADER = train_loader
        TEST_LOADER  = test_loader
        TOTAL_EPOCHS = epochs

        for epoch in range(TOTAL_EPOCHS):
            """
              Lists to store accuracy and loss for each model and each batch.
            """
            model1_train_pred = []
            model2_train_pred = []
            model3_train_pred = []

            model_actual_label = []

            model1_train_loss = []
            model2_train_loss = []
            model3_train_loss = []

            for (X_train_batch, y_train_batch) in TRAIN_LOADER:
                X_train_batch, y_train_batch = self._connect_GPU(
                    independent_data=X_train_batch, dependent_data=y_train_batch)

                # Do the prediction - train dataset
                model1, model2, model3 = self.classifier(X_train_batch)

                # Compute the models loss
                model1_loss = self._compute_model_loss(
                    model=model1, loss_function=self.model1_loss_function, actual_label=y_train_batch)
                model2_loss = self._compute_model_loss(
                    model=model2, loss_function=self.model2_loss_function, actual_label=y_train_batch)
                model3_loss = self._compute_model_loss(
                    model=model3, loss_function=self.model3_loss_function, actual_label=y_train_batch)

                # Do the backpropagation
                self._do_back_propagation(
                    optimizer=self.model1_optimizer, model_loss=model1_loss)
                self._do_back_propagation(
                    optimizer=self.model2_optimizer, model_loss=model2_loss)
                self._do_back_propagation(
                    optimizer=self.model3_optimizer, model_loss=model3_loss)

                # Compute the predicted labels
                model1_predicted = self._compute_predicted_label(
                    model_prediction=model1)
                model2_predicted = self._compute_predicted_label(
                    model_prediction=model2)
                model3_predicted = self._compute_predicted_label(
                    model_prediction=model3)

                # Compute the actual labels
                model_actual_label.extend(
                    self._compute_actual_label(actual_label=y_train_batch))

                # Store all preds and loss into list
                model1_train_pred.extend(model1_predicted)
                model2_train_pred.extend(model2_predicted)
                model3_train_pred.extend(model3_predicted)

                model1_train_loss.append(model1_loss.item())
                model2_train_loss.append(model2_loss.item())
                model3_train_loss.append(model3_loss.item())

            # Compute the accuracy
            model1_accuracy = self._model_accuracy(
                actual_label=model_actual_label, predicted_label=model1_train_pred)
            model2_accuracy = self._model_accuracy(
                actual_label=model_actual_label, predicted_label=model2_train_pred)
            model3_accuracy = self._model_accuracy(
                actual_label=model_actual_label, predicted_label=model3_train_pred)

            # Store accuracy & loss into the history
            self.history['m1_train_loss'].append(
                np.array(model1_train_loss).mean())
            self.history['m2_train_loss'].append(
                np.array(model2_train_loss).mean())
            self.history['m3_train_loss'].append(
                np.array(model3_train_loss).mean())

            self.history['m1_train_acc'].append(model1_accuracy)
            self.history['m2_train_acc'].append(model2_accuracy)
            self.history['m3_train_acc'].append(model3_accuracy)

            model1_test_pred = []
            model2_test_pred = []
            model3_test_pred = []

            model_actual_label = []

            model1_test_loss = []
            model2_test_loss = []
            model3_test_loss = []

            for (X_test_batch, y_test_batch) in TEST_LOADER:
                X_test_batch, y_test_batch = self._connect_GPU(
                    independent_data=X_test_batch, dependent_data=y_test_batch)
                # Do the prediction - test dataset
                model1, model2, model3 = self.classifier(X_test_batch)

                # Compute the models loss
                model1_loss = self._compute_model_loss(
                    model = model1, loss_function=self.model1_loss_function, actual_label=y_test_batch)
                model2_loss = self._compute_model_loss(
                    model = model2, loss_function=self.model2_loss_function, actual_label=y_test_batch)
                model3_loss = self._compute_model_loss(
                    model = model3, loss_function=self.model3_loss_function, actual_label=y_test_batch)

                # Compute the predicted labels
                model1_predicted = self._compute_predicted_label(
                    model_prediction = model1)
                model2_predicted = self._compute_predicted_label(
                    model_prediction = model2)
                model3_predicted = self._compute_predicted_label(
                    model_prediction = model3)

                # Compute the actual labels
                model_actual_label.extend(
                    self._compute_actual_label(actual_label=y_test_batch))

                # Store all preds and loss into list
                model1_test_pred.extend(model1_predicted)
                model2_test_pred.extend(model2_predicted)
                model3_test_pred.extend(model3_predicted)

                model1_test_loss.append(model1_loss.item())
                model2_test_loss.append(model2_loss.item())
                model3_test_loss.append(model3_loss.item())

            # Compute the accuracy
            model1_val_accuracy = self._model_accuracy(
                actual_label = model_actual_label, predicted_label=model1_test_pred)
            model2_val_accuracy = self._model_accuracy(
                actual_label = model_actual_label, predicted_label=model2_test_pred)
            model3_val_accuracy = self._model_accuracy(
                actual_label = model_actual_label, predicted_label=model3_test_pred)

            # Store accuracy & loss into the history
            self.history['m1_val_loss'].append(
                np.array(model1_test_loss).mean())
            self.history['m2_val_loss'].append(
                np.array(model2_test_loss).mean())
            self.history['m3_val_loss'].append(
                np.array(model3_test_loss).mean())

            self.history['m1_val_acc'].append(model1_val_accuracy)
            self.history['m2_val_acc'].append(model2_val_accuracy)
            self.history['m3_val_acc'].append(model3_val_accuracy)

            self._display(model1_train_loss = model1_train_loss,
                          model2_train_loss = model2_train_loss,
                          model3_train_loss = model3_train_loss,
                          model1_train_acc  = model1_accuracy,
                          model2_train_acc  = model2_accuracy,
                          model3_train_acc  = model3_accuracy,
                          model1_val_loss   = model1_test_loss,
                          model2_val_loss   = model2_test_loss,
                          model3_val_loss   = model3_test_loss,
                          model1_val_acc    = model1_val_accuracy,
                          model2_val_acc    = model2_val_accuracy,
                          model3_val_acc    = model3_val_accuracy,
                          running_epochs    = epoch,
                          total_epochs      = TOTAL_EPOCHS)
          
if __name__ == "__main__":
  trainer_ = Trainer(classifier = None, device = None)
