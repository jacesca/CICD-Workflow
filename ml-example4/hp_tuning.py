import json
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from utils_and_constants import (PROCESSED_DATASET,
                                 JSON_HYPERPARAMETERS_SCOPE,
                                 get_hp_tuning_results,
                                 load_data)
from utils_and_constants import JSON_BEST_PARAMS, MD_TUNNING_RESULTS


def main():
    X, y = load_data(PROCESSED_DATASET)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=1993)

    model = RandomForestClassifier()
    # Read the config file to define the hyperparameter search space
    param_grid = json.load(open(JSON_HYPERPARAMETERS_SCOPE, "r"))

    # Perform Grid Search Cross Validation on training data
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    print("====================Best Hyperparameters==================")
    print(json.dumps(best_params, indent=2))
    print("==========================================================")

    with open(JSON_BEST_PARAMS, "w") as outfile:
        json.dump(best_params, outfile)

    markdown_table = get_hp_tuning_results(grid_search)
    with open(MD_TUNNING_RESULTS, "w") as markdown_file:
        markdown_file.write("### Tunning Results\n\n")
        markdown_file.write(markdown_table)
        markdown_file.write('\n\n')

    # Save the results of hyperparameter tuning
    cv_results = pd.DataFrame(grid_search.cv_results_)
    markdown_table = cv_results.to_markdown(index=False)
    with open(MD_TUNNING_RESULTS, "a") as markdown_file:
        markdown_file.write("### Hyperparameter Run Output\n\n")
        markdown_file.write(markdown_table)
        markdown_file.write('\n')


if __name__ == "__main__":
    main()
