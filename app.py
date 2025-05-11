import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
from flask import Flask, request, render_template, jsonify
import traceback
import os

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

# Clean and prepare data
X = df[feature_columns].dropna()
y = df[target_columns].loc[X.index]
y_binary = (y <= 0.0005).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# =======================
#    MODEL PIPELINE
# =======================
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
    ('scaler', StandardScaler()),
    ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight='balanced')))
])

# =======================
#   HYPERPARAMETER TUNING
# =======================
param_grid = {
    'clf__estimator__C': [0.01, 0.1, 1, 10, 100],
    'clf__estimator__solver': ['lbfgs', 'liblinear'],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

# Save the best model
joblib.dump(grid_search.best_estimator_, 'model.pkl')

# Evaluate the model
y_pred = grid_search.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Evaluation Report:\n{classification_report(y_test, y_pred)}")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# =======================
#       FLASK ROUTES
# =======================

@app.route('/')


def home():
    """
    Renders the main HTML page.
    """
    return render_template("index.html")

@app.route('/health')
def health():
    return "Application is running!", 200
    
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests.
    Expects form data from the front end, makes predictions, and returns JSON.
    """
    try:
        # Extract form data and convert to float
        values = [float(request.form[col]) for col in feature_columns]
        
        # Create DataFrame for model input
        input_df = pd.DataFrame([values], columns=feature_columns)
        
        # Load the model
        model = joblib.load('model.pkl')
        
        # Get the probability of the positive class for each target (multi-output)
        probabilities = model.predict_proba(input_df)
        
        # Custom threshold for higher accuracy
        threshold = 0.5
        prediction = (probabilities[0] >= threshold).astype(int)

        # Check if any prediction is "Yes" (1)
        result = "Yes" if np.any(prediction == 1) else "No"
        
        # Return JSON response to frontend
        return jsonify({'prediction': result, 'accuracy': accuracy * 100})
    
    except Exception as e:
        print("Prediction Error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# =======================
#   RUN THE APPLICATION
# =======================



if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Get the port from the environment variable
    print(f"🚀 Application starting on port {port}...")  # Log the port number
    app.run(host='0.0.0.0', port=port, debug=True)





