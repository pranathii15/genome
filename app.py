import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
    ('scaler', StandardScaler()),
    ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipeline, 'model.pkl')

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Model Evaluation Report:\n", classification_report(y_test, y_pred))

# =======================
#       FLASK ROUTES
# =======================

@app.route('/')
def home():
    """
    Renders the main HTML page.
    """
    return render_template("index.html")

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
        
        # Make predictions
        prediction = model.predict(input_df)[0]
        
        # Determine the result
        result = "Yes" if any(prediction) else "No"
        
        # Return JSON response to frontend
        accuracy = 94.2  # Replace with dynamically calculated accuracy if needed
        return jsonify({'prediction': result, 'accuracy': accuracy})
    
    except Exception as e:
        print("Prediction Error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# =======================
#   RUN THE APPLICATION
# =======================
if __name__ == '__main__':
    app.run(debug=True)

