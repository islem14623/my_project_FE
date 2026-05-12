from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return "IDS API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = data["features"]

    # fake prediction for testing
    prediction = 1 if sum(features) > 10 else 0

    return jsonify({
        "prediction": prediction,
        "result": "ATTACK" if prediction == 1 else "NORMAL"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
