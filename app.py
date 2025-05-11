import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
from flask import Flask, request, render_template, jsonify
import traceback

app = Flask(__name__)

# =======================
#       LOAD DATA
# =======================
df = pd.read_csv("CMK.hg19.AllInteractions.SP4.FDR0.001.xls", sep="\t")

# Define feature and target columns
target_columns = [
    'CG1_p_value', 'CG2_p_value', 'CC1_p_value',
    'CC2_p_value', 'CN1_p_value', 'CN2_p_value'
]

feature_columns = [
    'Feature_Start', 'Interactor_Start', 'Interactor_End', 'distance',
    'CG1_SuppPairs', 'CG2_SuppPairs', 'CC1_SuppPairs', 'CC2_SuppPairs',
    'CN1_SuppPairs', 'CN2_SuppPairs',
    'Normal', 'CarboplatinTreated', 'GemcitabineTreated'
]

# =======================
#   CLEAN AND SPLIT DATA
# =======================
def clean_and_split_data():
    X = df[feature_columns].dropna()
    y = df[target_columns].loc[X.index]
    y_binary = (y <= 0.0005).astype(int)  # Binary conversion based on threshold

    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# =======================
#    MODEL PIPELINE
# =======================
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
    ])

    # =======================
    #     HYPERPARAMETER TUNING
    # =======================
    param_grid = {
        'clf__estimator__C': [0.01, 0.1, 1, 10],  # Regularization strength
        'clf__estimator__solver': ['lbfgs', 'liblinear'],  # Solvers to try
        'clf__estimator__penalty': ['l2'],  # Regularization type
    }

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

    # Train the model with hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Best hyperparameters found
    print(f"Best hyperparameters: {grid_search.best_params_}")

    # Save the best model
    joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
    
    return grid_search

# =======================
#     EVALUATE THE MODEL
# =======================
def evaluate_model(grid_search, X_test, y_test):
    y_pred = grid_search.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Detailed classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))

# =======================
#     FLASK APP
# =======================
@app.route('/')
def index():
    return "Model Training and Evaluation is Complete!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Example of how to predict with the model
        data = request.get_json()
        model = joblib.load('best_model.pkl')
        # You would need to extract features from `data` and predict.
        prediction = model.predict(data)
        return jsonify(prediction.tolist())

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

def main():
    # Step 1: Clean and split data
    X_train, X_test, y_train, y_test = clean_and_split_data()

    # Step 2: Train the model
    grid_search = train_model(X_train, y_train)

    # Step 3: Evaluate the model
    evaluate_model(grid_search, X_test, y_test)

if __name__ == '__main__':
    main()  # Call main function to run model training and evaluation
    app.run(debug=True)  # Start Flask app

