import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay,
                             accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List


def target_encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    target_column: str
) -> pd.DataFrame:
    """Data preparation: target encoding"""
    encoded_data = df.copy()

    # Iterate through categorical columns
    for col in categorical_columns:
        # Calculate mean target value for each category
        encoding_map = df.groupby(col)[target_column].mean().to_dict()
        # Apply target encoding
        encoded_data[col] = encoded_data[col].map(encoding_map)
    return encoded_data


def impute_and_scale_data(df_features: pd.DataFrame) -> pd.DataFrame:
    """Imputing and Scaling"""
    # Impute data with mean strategy
    imputer = SimpleImputer(strategy="mean")
    X_preprocessed = imputer.fit_transform(df_features.values)
    # Scale and fit with zero mean and unit variance
    scaler = StandardScaler()
    X_preprocessed = scaler.fit_transform(X_preprocessed)
    return pd.DataFrame(X_preprocessed, columns=df_features.columns)


# Global vars
SEED = 42
TARGET_COL = 'RainTomorrow'
CATEGO_COLS = ['Location', 'WindGustDir', 'WindDir9am',
               'WindDir3pm', 'RainToday', ]

# Read data
df = pd.read_csv('data-sources/weather.csv',
                 parse_dates=['Date'],
                 index_col='Date',
                 dtype={'RainTomorrow': bool},
                 true_values=['Yes'], false_values=['No'])
df['month'] = df.index.month

# Encode categorical variables
df_encoded = target_encode_categorical_features(
    df, CATEGO_COLS, TARGET_COL
)

# Split into features and target
X = df_encoded.drop(columns=TARGET_COL)
y = df_encoded[TARGET_COL]

# Impute missing values
X_imputed = impute_and_scale_data(X)
assert X_imputed.isnull().sum().sum() == 0

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                    random_state=SEED)

# Random Forest Classifier
model = RandomForestClassifier(max_depth=2, n_estimators=50,
                               random_state=SEED)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

result_text = f'''
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
    F1: {f1}
'''
with open('ml-example/result.txt', 'w') as f:
    f.write(result_text)

ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test, cmap=plt.cm.Blues
)
plt.savefig('ml-example/confusion-matrix.png')
