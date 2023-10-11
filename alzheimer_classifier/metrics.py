from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import torch
import matplotlib.pyplot as plt

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

    for (X_batch, y_batch) in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.long()
        y_batch = y_batch.to(device)

        model1_prediction, model2_prediction, model3_prediction = model(X_batch)

        model1_prediction = torch.argmax(model1_prediction, dim=1)
        model2_prediction = torch.argmax(model2_prediction, dim=1)
        model3_prediction = torch.argmax(model3_prediction, dim=1)

        model1_prediction = model1_prediction.cpu().detach().flatten().numpy()
        model2_prediction = model2_prediction.cpu().detach().flatten().numpy()
        model3_prediction = model3_prediction.cpu().detach().flatten().numpy()

        majority_voting = _compute_majority_voting(predicted_label=zip(model1_prediction, model2_prediction, model3_prediction))

        predict_labels.extend(majority_voting)
        actual_labels.extend(y_batch.cpu().detach().flatten().numpy())

    return actual_labels, predict_labels

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
    sns.heatmap(confusion_matrix(actual_labels, predict_labels), annot=True, fmt=".1f")
    plt.show()

def model_performance(model=None, train_loader=None, test_loader=None, device = None):
    """
    Compute and display the performance metrics of a model on both training and testing datasets.

    Args:
        model: The model to evaluate.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the testing dataset.
        device: The device (e.g., CPU or GPU) to use for evaluation.
    """
    actual_train_labels, predict_train_labels = _compute_performance(model = model, dataloader = train_loader, device = device)

    print("Evaluation of Train Dataset with {} records.".format(len(actual_train_labels)), '\n')

    print("ACCURACY  # {} ".format(accuracy_score(actual_train_labels, predict_train_labels)))
    print("PRECISION # {} ".format(recall_score(actual_train_labels, predict_train_labels, average='macro')))
    print("RECALL    # {} ".format(precision_score(actual_train_labels, predict_train_labels, average='macro')))
    print("F1_SCORE  # {} ".format(f1_score(actual_train_labels, predict_train_labels, average='macro')))

    print("_" * 50, "\n")

    actual_train_labels, predict_train_labels = _compute_performance(model=model, dataloader=test_loader, device = device)

    print("Ëvaluation of Test Dataset  {} records.".format(len(actual_train_labels)), '\n')

    print("ACCURACY  # {} ".format(accuracy_score(actual_train_labels, predict_train_labels)))
    print("PRECISION # {} ".format(recall_score(actual_train_labels, predict_train_labels, average='macro')))
    print("RECALL    # {} ".format(precision_score(actual_train_labels, predict_train_labels, average='macro')))
    print("F1_SCORE  # {} ".format(f1_score(actual_train_labels, predict_train_labels, average='macro')))

    print("_" * 50, "\n")

    print("Classification report for test dataset\n")
    _show_classification_report(actual_labels=actual_train_labels, predict_labels=predict_train_labels)

    print("Confusion matrix for test dataset\n")
    _confusion_matrix(actual_labels=actual_train_labels, predict_labels=predict_train_labels)