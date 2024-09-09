import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from utils_and_constants import (PNG_CONFUSION_IMAGE,
                                 JSON_METRICS,
                                 CSV_ROC_CURVE,
                                 CSV_TEST_PREDICTIONS)


def plot_confusion_matrix(model, X_test, y_test):
    _ = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                              cmap=plt.cm.Blues)
    plt.savefig(PNG_CONFUSION_IMAGE)


def save_metrics(metrics):
    with open(JSON_METRICS, "w") as fp:
        json.dump(metrics, fp)


def save_predictions(y_test, y_pred):
    # Store predictions data for confusion matrix
    cdf = pd.DataFrame(
        np.column_stack([y_test, y_pred]),
        columns=["true_label", "predicted_label"]
    ).astype(int)
    cdf.to_csv(CSV_TEST_PREDICTIONS, index=None)


def save_roc_curve(y_test, y_pred_proba):
    # Calcualte ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    # Store roc curve data
    cdf = pd.DataFrame(
        np.column_stack([fpr, tpr]),
        columns=["fpr", "tpr"]
    ).astype(float)
    cdf.to_csv(CSV_ROC_CURVE, index=None)
