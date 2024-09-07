import shutil
from pathlib import Path

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["Date"]
TARGET_COLUMN = "RainTomorrow"
RAW_DATASET = "ml-example3/raw-data/weather.csv"
PROCESSED_DATASET = "ml-example3/processed-data/weather.csv"
RFC_FOREST_DEPTH = 2

PNG_CONFUSION_IMAGE = "ml-example3/evaluation-result/confusion_matrix.png"
JSON_METRICS = "ml-example3/evaluation-result/metrics.json"
CSV_TEST_PREDICTIONS = "ml-example3/evaluation-result/predictions.csv"
CSV_ROC_CURVE = "ml-example3/evaluation-result/roc_curve.csv"


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(e)
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)
