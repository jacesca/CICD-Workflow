# Prepare the yaml file
yml_str = """stages:
  preprocess:
    cmd: python ml-example3/preprocess_data.py
    deps:
    - ml-example3/raw-data/weather.csv
    - ml-example3/preprocess_data.py
    - ml-example3/.env
    - ml-example3/utils_and_constants.py
    outs:
    - ml-example3/processed-data/weather.csv
  train:
    cmd: python ml-example3/train.py
    deps:
    - ml-example3/metrics_and_plots.py
    - ml-example3/model.py
    - ml-example3/processed-data/weather.csv
    - ml-example3/train.py
    - ml-example3/.env
    - ml-example3/utils_and_constants.py
    outs:
    - ml-example3/evaluation-result/confusion_matrix.png
    metrics:
      - ml-example3/evaluation-result/metrics.json:
          cache: false
    plots:
      - ml-example3/evaluation-result/predictions.csv:
          template: confusion_normalized
          x: predicted_label
          y: true_label
          x_label: 'Predicted label'
          y_label: 'True label'
          title: Confusion matrix
          cache: false
"""
with open("dvc.yaml", "w") as f:
    f.write(yml_str)
