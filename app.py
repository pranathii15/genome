import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import traceback
import os

app = Flask(__name__)
CORS(app)  

print("Loading data and training model...")
df = pd.read_csv("CMK.hg19.AllInteractions.SP4.FDR0.001.xls", sep="\t")

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

X = df[feature_columns].dropna()
y = df[target_columns].loc[X.index]
y_binary = (y <= 0.0005).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'model.pkl')
model = joblib.load('model.pkl')

y_pred = pipeline.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred) * 100
print("Model Evaluation Report:\n", classification_report(y_test, y_pred))
print(f"Model Accuracy: {model_accuracy:.2f}%")


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
        
        # Make predictions
        prediction = model.predict(input_df)[0]
        
        # Determine the result
        result = "Yes" if any(prediction) else "No"
        
        # Return JSON response to frontend
        return jsonify({
            'prediction': result,
            'accuracy': f"{model_accuracy:.2f}"
        })
    
    except Exception as e:
        print("Prediction Error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000



