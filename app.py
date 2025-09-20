from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)


model = joblib.load("crop_rf_model.pkl")
scaler = joblib.load("scaler.pkl")


crop_costs = {
    'rice': 5000,
    'wheat': 3000,
    'maize': 4000,
    'cotton': 7000,
    'sugarcane': 6000,
}


@app.route('/recommend', methods=['POST'])
def recommend_crop():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        N = float(data.get('nitrogen', 0))
        P = float(data.get('phosphorus', 0))
        K = float(data.get('potassium', 0))
        rainfall = float(data.get('rainfall', 0))
        ph = float(data.get('ph', 0))

        sample = pd.DataFrame([{
            'N_SOIL': N,
            'P_SOIL': P,
            'K_SOIL': K,
            'TEMPERATURE': 23,
            'HUMIDITY': 70,
            'ph': ph,
            'RAINFALL': rainfall
        }])

        sample_scaled = scaler.transform(sample)

        probabilities = model.predict_proba(sample_scaled)[0]
        classes = model.classes_

        crop_probs = zip(classes, probabilities)
        sorted_crops = sorted(crop_probs, key=lambda x: x[1], reverse=True)[:4]

        results = []
        for crop, prob in sorted_crops:
            cost = crop_costs.get(crop.lower(), 0)
            results.append({
                'crop': crop,
                'confidence': int(prob * 100),
                'cost': cost,
                'yield': "n/a"
            })

        return jsonify({'recommendations': results})

    except Exception as e:

        print("Error in /recommend:", e)
        return jsonify({"error": "Server error: " + str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
