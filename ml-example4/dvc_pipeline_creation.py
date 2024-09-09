# Prepare the yaml file
yml_str = """stages:
  preprocess:
    cmd: python ml-example4/preprocess_dataset.py
    deps:
    - ml-example4/raw-data/weather.csv
    - ml-example4/preprocess_dataset.py
    - ml-example4/.env
    - ml-example4/utils_and_constants.py
    outs:
    - ml-example4/processed-data/weather.csv
  hp_tune:
    cmd: python ml-example4/hp_tuning.py
    deps:
    - ml-example4/processed-data/weather.csv
    - ml-example4/config/hp_config.json
    - ml-example4/hp_tuning.py
    outs:
    - ml-example4/evaluation-result/rfc_best_params.json
    - ml-example4/evaluation-result/hp_tuning_results.md:
        cache: false
  train:
    cmd: python ml-example4/train.py
    deps:
    - ml-example4/metrics_and_plots.py
    - ml-example4/model.py
    - ml-example4/processed-data/weather.csv
    - ml-example4/evaluation-result/rfc_best_params.json
    - ml-example4/train.py
    - ml-example4/.env
    - ml-example4/utils_and_constants.py
    outs:
    - ml-example4/evaluation-result/confusion_matrix.png
    metrics:
      - ml-example4/evaluation-result/metrics.json:
          cache: false
    plots:
      - ml-example4/evaluation-result/predictions.csv:
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
