import pickle
import os
import logging
from flask import Flask, request, jsonify

# Configuration
model_file = os.environ.get('MODEL_FILE', 'decision_tree_HostRangeClassifier_v1.bin')
port = int(os.environ.get('PORT', 9696))

logging.basicConfig(level=logging.INFO)

# Load model
try:
    with open(model_file, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    logging.info(f"Model {model_file} loaded successfully. Ready to recieve requests and make predictions.")
except FileNotFoundError:
    logging.error(f"Model file {model_file} not found.")
    exit(1)

app = Flask('strain-classification')

@app.route('/predict_host_range', methods=['POST'])
def predict_host_range():
    """
    Predicts the host range of a Salmonella strain based on its nutrient-utilization profile.
    """
    try:
        strain_profile = request.get_json()

        if not strain_profile:
            raise ValueError("No input data provided")

        X = dv.transform([strain_profile])
        y_pred = model.predict_proba(X)[0, 1]

        prediction = 'Generalist' if y_pred >= 0.5 else 'Specialist'
        result = {
            'Generalist probability': float(y_pred),
            'Host range': prediction
        }

        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)