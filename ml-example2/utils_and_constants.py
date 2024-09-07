import shutil
from pathlib import Path

ROOT = 'ml-example2'
DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["Date"]
TARGET_COLUMN = "RainTomorrow"
RAW_DATASET = "ml-example2/raw-data/weather.csv"
PROCESSED_DATASET = "ml-example2/processed-data/weather.csv"

CONFUSION_IMAGE = "ml-example2/evaluation-result/confusion_matrix.png"
JSON_METRICS = "ml-example2/evaluation-result/metrics.json"


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(e)
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)
