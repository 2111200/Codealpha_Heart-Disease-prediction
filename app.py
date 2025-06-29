from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "heart_disease_model.pkl"
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Define feature names
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Home Page
@app.route("/")
def index():
    return render_template("index.html")

# Handle Form Submission
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values
        form_data = [float(request.form[feature]) for feature in FEATURES]

        # Convert to NumPy array & reshape
        input_data = np.array(form_data).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)[0]  # 0: No disease, 1: Disease

        # Show the result
        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return f"Error: {e}"

# API Endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)