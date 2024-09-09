import json
import shutil
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv


load_dotenv("ml-example4/.env")


RFC_FOREST_DEPTH = int(os.getenv('RFC_FOREST_DEPTH'))

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["Date"]
TARGET_COLUMN = "RainTomorrow"
RAW_DATASET = "ml-example4/raw-data/weather.csv"
PROCESSED_DATASET = "ml-example4/processed-data/weather.csv"

JSON_HYPERPARAMETERS_SCOPE = "ml-example4/config/hp_config.json"

PNG_CONFUSION_IMAGE = "ml-example4/evaluation-result/confusion_matrix.png"
JSON_METRICS = "ml-example4/evaluation-result/metrics.json"
CSV_TEST_PREDICTIONS = "ml-example4/evaluation-result/predictions.csv"
CSV_ROC_CURVE = "ml-example4/evaluation-result/roc_curve.csv"

JSON_BEST_PARAMS = "ml-example4/evaluation-result/rfc_best_params.json"
MD_TUNNING_RESULTS = "ml-example4/evaluation-result/hp_tuning_results.md"


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(e)
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def load_hyperparameters(hyperparameter_file):
    with open(hyperparameter_file, "r") as json_file:
        hyperparameters = json.load(json_file)
    return hyperparameters


def get_hp_tuning_results(grid_search: GridSearchCV) -> str:
    """Get the results of hyperparameter tuning in a Markdown table"""
    cv_results = pd.DataFrame(grid_search.cv_results_)

    # Extract and split the 'params' column into subcolumns
    params_df = pd.json_normalize(cv_results["params"])

    # Concatenate the params_df with the original DataFrame
    cv_results = pd.concat([cv_results, params_df], axis=1)

    # Get the columns to display in the Markdown table
    cv_results = cv_results[
        ["rank_test_score", "mean_test_score", "std_test_score"]
        + list(params_df.columns)
    ]

    cv_results.sort_values(by="mean_test_score", ascending=False, inplace=True)
    return cv_results.to_markdown(index=False)
