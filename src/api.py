from flask import Flask, request, jsonify
import numpy as np
import os

# Only load model/scaler if files exist (for production)
# During CI/CD tests, these files won't exist
MODEL_PATH = "models/final_pso_cnn_model.keras"
SCALER_PATH = "models/pso_scaler.pkl"
FEATURES_PATH = "models/pso_selected_features.pkl"

# Check if we're in production (model files exist) or testing
PRODUCTION_MODE = os.path.exists(MODEL_PATH)

if PRODUCTION_MODE:
    import joblib
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selected_features = joblib.load(FEATURES_PATH)
    print("✅ Model loaded successfully!")
else:
    model = None
    scaler = None
    selected_features = None
    print("⚠️ Running in TEST mode (model files not found)")

app = Flask(__name__)

@app.route("/")
def home():
    status = "PRODUCTION" if PRODUCTION_MODE else "TEST"
    return f"IDS API is running (PSO + CNN model) - Mode: {status}"

@app.route("/predict", methods=["POST"])
def predict():
    if not PRODUCTION_MODE:
        return jsonify({"error": "API running in test mode - model not loaded"}), 503
    
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    features = np.array(data["features"], dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.shape[1] < int(np.max(selected_features)) + 1:
        return jsonify({"error": f"Expected at least {int(np.max(selected_features)) + 1} features"}), 400

    features_selected = features[:, selected_features]
    features_scaled = scaler.transform(features_selected)
    features_cnn = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
    pred_prob = model.predict(features_cnn, verbose=0)

    pred = (pred_prob > 0.5).astype(int).flatten()
    return jsonify({
        "prediction": int(pred[0]),
        "result": "ATTACK" if pred[0] == 1 else "NORMAL",
        "confidence": round(float(pred_prob[0][0]), 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
