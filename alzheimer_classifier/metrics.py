from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def _compute_majority_voting(predicted_label=None):
    """
    Compute majority voting for a list of predictions from three models.

    Args:
        predicted_label (list): A list of predictions from three models.

    Returns:
        list: A list of majority voting predictions.
    """
    voting_predict_labels = []

    for model1_pred, model2_pred, model3_pred in predicted_label:
        majority_voting = [model1_pred, model2_pred, model3_pred]

        majority_count = Counter(majority_voting)
        most_common_value = max(majority_count, key=majority_count.get)
        voting_predict_labels.append(most_common_value)

    return voting_predict_labels


def _compute_performance(model=None, dataloader=None, device = None):
    """
    Compute the performance of a model on a given data loader.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the dataset.

    Returns:
        tuple: A tuple containing actual labels and predicted labels.
    """
    predict_labels = []
    actual_labels = []
    IMAGE_ARRAY = []

    for (X_batch, y_batch) in dataloader:
        IMAGE_ARRAY.append(X_batch)
        X_batch = X_batch.to(device)
        X_batch = X_batch.float()
        y_batch = y_batch.long()
        y_batch = y_batch.to(device)

        model1_prediction, model2_prediction, model3_prediction = model(
            X_batch)

        model1_prediction = torch.argmax(model1_prediction, dim=1)
        model2_prediction = torch.argmax(model2_prediction, dim=1)
        model3_prediction = torch.argmax(model3_prediction, dim=1)

        model1_prediction = model1_prediction.cpu().detach().flatten().numpy()
        model2_prediction = model2_prediction.cpu().detach().flatten().numpy()
        model3_prediction = model3_prediction.cpu().detach().flatten().numpy()

        majority_voting = _compute_majority_voting(predicted_label=zip(
            model1_prediction, model2_prediction, model3_prediction))

        predict_labels.extend(majority_voting)
        actual_labels.extend(y_batch.cpu().detach().flatten().numpy())

    return IMAGE_ARRAY, actual_labels, predict_labels


def _show_classification_report(actual_labels=None, predict_labels=None):
    """
    Show the classification report based on actual and predicted labels.

    Args:
        actual_labels: Actual ground truth labels.
        predict_labels: Predicted labels.
    """
    print(classification_report(actual_labels, predict_labels))


def _confusion_matrix(actual_labels=None, predict_labels=None):
    """
    Display a confusion matrix based on actual and predicted labels.

    Args:
        actual_labels: Actual ground truth labels.
        predict_labels: Predicted labels.
    """
    sns.heatmap(confusion_matrix(actual_labels, predict_labels),
                annot=True, fmt=".1f")
    plt.show()


def _plot_test_prediction(IMAGE=None, actual_labels=None, predict_labels=None):
    """
        Display a grid of sample images and their corresponding labels.

        This method shows a grid of sample images from the training data and their corresponding labels.
        It can be useful for visually inspecting a subset of the dataset.

        Returns:
            None
    """
    IMAGE = IMAGE.reshape(IMAGE.shape[0], 120, 120, 3)

    plt.figure(figsize=(12, 8))

    for index, image in enumerate(IMAGE):
        plt.subplot(4, 5, index + 1)
        try:
            plt.imshow(image)
            plt.title('Actual: {} \n Predicted: {}'.format(
                'Mild' if actual_labels[index] == 0 else 'Moderate' if actual_labels[
                    index] == 1 else 'No' if actual_labels[index] == 2 else 'Very Mild',
                'Mild' if predict_labels[index] == 0 else 'Moderate' if predict_labels[
                    index] == 1 else 'No' if predict_labels[index] == 2 else 'Very Mild'
            ))

            plt.tight_layout()
            plt.axis("off")
        except Exception as e:
            print("Exception caught {} ".format(e))

    plt.show()


def model_performance(model=None, train_loader=None, test_loader=None, device=None):
    """
    Compute and display the performance metrics of a model on both training and testing datasets.

    Args:
        model: The model to evaluate.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the testing dataset.
        device: The device (e.g., CPU or GPU) to use for evaluation.
    """
    IMAGE, actual_train_labels, predict_train_labels = _compute_performance(
        model=model, dataloader=train_loader, device=device)

    print("Evaluation of Train Dataset with {} records.".format(
        len(actual_train_labels)), '\n')

    print("ACCURACY  # {} ".format(accuracy_score(
        actual_train_labels, predict_train_labels)))
    print("PRECISION # {} ".format(recall_score(
        actual_train_labels, predict_train_labels, average='macro')))
    print("RECALL    # {} ".format(precision_score(
        actual_train_labels, predict_train_labels, average='macro')))
    print("F1_SCORE  # {} ".format(
        f1_score(actual_train_labels, predict_train_labels, average='macro')))

    print("_" * 50, "\n")

    IMAGE, actual_test_labels, predict_test_labels = _compute_performance(
        model=model, dataloader=test_loader)

    print("Ëvaluation of Test Dataset  {} records.".format(
        len(actual_train_labels)), '\n')

    print("ACCURACY  # {} ".format(accuracy_score(
        actual_test_labels, predict_test_labels)))
    print("PRECISION # {} ".format(recall_score(
        actual_test_labels, predict_test_labels, average='macro')))
    print("RECALL    # {} ".format(precision_score(
        actual_test_labels, predict_test_labels, average='macro')))
    print("F1_SCORE  # {} ".format(
        f1_score(actual_test_labels, predict_test_labels, average='macro')))

    print("_" * 50, "\n")

    print("Classification report for test dataset\n")
    _show_classification_report(
        actual_labels=actual_train_labels, predict_labels=predict_train_labels)

    print("Confusion matrix for test dataset\n")
    _confusion_matrix(actual_labels=actual_train_labels,
                      predict_labels=predict_train_labels)

    _plot_test_prediction(
        IMAGE=IMAGE[0][0:20], actual_labels=actual_test_labels[0:20], predict_labels=predict_test_labels[0:20])
