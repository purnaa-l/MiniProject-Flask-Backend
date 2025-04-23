# from flask import Flask, request, jsonify
# import joblib
# import os
# from flask_cors import CORS
# app = Flask(__name__)

# # Allow only the frontend's origin (adjust for your setup)
# CORS(app, resources={r"/predict": {"origins": "http://localhost:5173", "methods": ["POST"], "allow_headers": ["Content-Type"]}})
# CORS(app)
# # Load model and scaler with proper absolute paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
# scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))


# @app.route("/predict", methods=["POST", "OPTIONS"])
# def predict():
#     if request.method == 'OPTIONS':
#         # This is a preflight request, return a 200 response.
#         return '', 200

#     try:
#         # Ensure the request has JSON data
#         if request.is_json:
#             data = request.get_json()
#               # Parse the JSON body
            
#             temperature = data.get('temperatureCelsius', 0.0) or data.get('temperature_celsius', 0.0)
#             wind_kph = data.get('windKph', 0.0) or data.get('wind_kph', 0.0)
#             pressure = data.get('pressureMb', 0.0) or data.get('pressure_mb', 0.0)
#             humidity = data.get('humidity', 0.0)
#             cloud = data.get('cloud', 0.0)
#             uv_index = data.get('uvIndex', 0.0) or data.get('uv_index', 0.0)
#             features = [
#                 data["temperature_celsius"],
#                 data["wind_kph"],
#                 data["pressure_mb"],
#                 data["humidity"],
#                 data["cloud"],
#                 data["uv_index"]
#             ]

#             # Scale and predict
#             scaled_features = scaler.transform([features])
#             prediction = model.predict(scaled_features)[0]
#             return jsonify({"prediction": round(prediction, 2)})

#         else:
#             return jsonify({"error": "Request must be JSON"}), 415

#     except Exception as e:
#         print("🔥 Error during prediction:", e)
#         return jsonify({"error": "Prediction failed. Try again."}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS
app = Flask(__name__)

# Allow only the frontend's origin (adjust for your setup)
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173", "methods": ["POST"], "allow_headers": ["Content-Type"]}})
CORS(app)
# Load model and scaler with proper absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == 'OPTIONS':
        # This is a preflight request, return a 200 response.
        return '', 200

    try:
        # Ensure the request has JSON data
        if request.is_json:
            data = request.get_json()
              # Parse the JSON body
            
            temperatureCelsius = data.get('temperatureCelsius', 0.0) or data.get('temperature_celsius', 0.0)
            windKph = data.get('windKph', 0.0) or data.get('wind_kph', 0.0)
            pressureMb = data.get('pressureMb', 0.0) or data.get('pressure_mb', 0.0)
            humidity = data.get('humidity', 0.0)
            cloud = data.get('cloud', 0.0)
            uvIndex = data.get('uvIndex', 0.0) or data.get('uv_index', 0.0)
            features = [
                data["temperatureCelsius"],
                data["windKph"],
                data["pressureMb"],
                data["humidity"],
                data["cloud"],
                data["uvIndex"]
            ]

            # Scale and predict
            scaled_features = scaler.transform([features])
            prediction = model.predict(scaled_features)[0]
            return jsonify({"prediction": round(prediction, 2)})

        else:
            return jsonify({"error": "Request must be JSON"}), 415

    except Exception as e:
        print("🔥 Error during prediction:", e)
        return jsonify({"error": "Prediction failed. Try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)