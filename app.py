from flask import Flask, render_template, request
import pickle
import numpy as np
from feature import generate_data_set  # import the feature extraction
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load trained model and scaler
try:
    with open("pickle/model.pkl", "rb") as file:
        model_data = pickle.load(file)
        model = model_data['model']
        scaler = model_data['scaler']
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return render_template("index.html", prediction_text="Model not loaded. Please check the model file.", url="", color="red")
    
    url = request.form["url"]

    # Extract features using your feature.py script
    try:
        input_features = generate_data_set(url)
        
        # Check if we have the right number of features
        if len(input_features) != model.n_features_in_:
            return render_template("index.html", 
                                 prediction_text=f"Error: Expected {model.n_features_in_} features but got {len(input_features)}", 
                                 url=url, 
                                 color="red")
        
        # Scale the features
        input_features = np.array(input_features).reshape(1, -1)
        input_features = scaler.transform(input_features)

        # Prediction + probability
        prediction = model.predict(input_features)[0]
        proba = model.predict_proba(input_features)[0]
        
        # Get the probability for the predicted class
        confidence = proba[prediction]

        if prediction == 1:  # Assuming 1 is legitimate
            result = f"Legitimate Website ✅ ({confidence*100:.2f}% confidence)"
            color = "green"
        else:
            result = f"Phishing / Malicious Website ❌ ({confidence*100:.2f}% confidence)"
            color = "red"
            
    except Exception as e:
        result = f"Error processing URL: {str(e)}"
        color = "orange"

    return render_template("index.html", prediction_text=result, url=url, color=color)

if __name__ == "__main__":
    app.run(debug=True)