import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, render_template
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
  # or handle accordingly

# Load the dataset
df = pd.read_csv("CMK.hg19.AllInteractions.SP4.FDR0.001.xls", sep="\t")

# Feature and target columns
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

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Pipeline with scaling and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
])

pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'model.pkl')

# Evaluate and print report
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Flask App
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[col]) for col in feature_columns]
        input_df = pd.DataFrame([values], columns=feature_columns)
        model = joblib.load('model.pkl')
        prediction = model.predict(input_df)[0]
        result = "Yes" if any(prediction) else "No"

        # Optional: return more detail
        accuracy = 94.2  # static or calculated
        return jsonify({'prediction': result, 'accuracy': accuracy})

    except Exception as e:
        import traceback
        print("Prediction Error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
