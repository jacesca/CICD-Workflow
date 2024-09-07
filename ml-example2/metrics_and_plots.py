import json

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils_and_constants import CONFUSION_IMAGE, JSON_METRICS


def plot_confusion_matrix(model, X_test, y_test):
    _ = ConfusionMatrixDisplay.from_estimator(model,
                                              X_test, y_test,
                                              cmap=plt.cm.Blues)
    plt.savefig(CONFUSION_IMAGE)


def save_metrics(metrics):
    with open(JSON_METRICS, "w") as fp:
        json.dump(metrics, fp)
