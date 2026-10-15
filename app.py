import os
import logging
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Directory resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Global variables for ML artifacts
model = None
scaler = None
expected_columns = None
model_load_error = None

def load_ml_artifacts():
    """Load machine learning model, scaler, and column definitions."""
    global model, scaler, expected_columns, model_load_error
    try:
        def get_path(filename):
            primary = os.path.join(MODELS_DIR, filename)
            fallback = os.path.join(BASE_DIR, filename)
            if os.path.exists(primary):
                return primary
            elif os.path.exists(fallback):
                return fallback
            else:
                raise FileNotFoundError(f"Required artifact '{filename}' not found in models/ or root directory.")

        model_path = get_path("Logistic_Regression.pkl")
        scaler_path = get_path("scaler.pkl")
        columns_path = get_path("columns.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        expected_columns = joblib.load(columns_path)
        
        logger.info(f"Loaded ML Model from {model_path}")
        logger.info(f"Loaded Scaler from {scaler_path}")
        logger.info(f"Loaded {len(expected_columns)} expected feature columns from {columns_path}")
        model_load_error = None
        return True
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Failed to load ML artifacts: {model_load_error}")
        return False

# Load artifacts on startup
load_ml_artifacts()

def validate_and_parse_input(data):
    """Validate and parse 11 user-facing input health fields against reasonable clinical boundaries."""
    errors = []
    parsed = {}

    def parse_int(val, name, min_val, max_val):
        try:
            v = int(val)
            if not (min_val <= v <= max_val):
                errors.append(f"'{name}' must be between {min_val} and {max_val}.")
            return v
        except (ValueError, TypeError):
            errors.append(f"Please enter a valid integer for '{name}'.")
            return None

    def parse_float(val, name, min_val, max_val):
        try:
            v = float(val)
            if not (min_val <= v <= max_val):
                errors.append(f"'{name}' must be between {min_val} and {max_val}.")
            return v
        except (ValueError, TypeError):
            errors.append(f"Please enter a valid numeric value for '{name}'.")
            return None

    def parse_choice(val, name, allowed):
        if str(val).strip().upper() not in [a.upper() for a in allowed]:
            errors.append(f"Please select a valid option for '{name}'. Allowed: {', '.join(allowed)}.")
            return None
        return str(val).strip()

    # Reasonable clinical input boundaries (11 user-facing inputs)
    parsed['Age'] = parse_int(data.get('age'), 'Age', 18, 120)
    parsed['RestingBP'] = parse_int(data.get('resting_bp'), 'Resting Blood Pressure', 60, 240)
    parsed['Cholesterol'] = parse_int(data.get('cholesterol'), 'Serum Cholesterol', 0, 600)
    parsed['FastingBS'] = parse_int(data.get('fasting_bs'), 'Fasting Blood Sugar', 0, 1)
    parsed['MaxHR'] = parse_int(data.get('max_hr'), 'Maximum Heart Rate', 40, 220)
    parsed['Oldpeak'] = parse_float(data.get('oldpeak'), 'Oldpeak (ST Depression)', -2.6, 10.0)

    sex_val = parse_choice(data.get('sex'), 'Sex', ['M', 'F'])
    if sex_val:
        parsed['Sex'] = sex_val.upper()

    chest_pain_val = parse_choice(data.get('chest_pain'), 'Chest Pain Type', ['ATA', 'NAP', 'TA', 'ASY'])
    if chest_pain_val:
        parsed['ChestPainType'] = chest_pain_val.upper()

    resting_ecg_val = parse_choice(data.get('resting_ecg'), 'Resting ECG', ['Normal', 'ST', 'LVH'])
    if resting_ecg_val:
        parsed['RestingECG'] = resting_ecg_val

    exercise_angina_val = parse_choice(data.get('exercise_angina'), 'Exercise Angina', ['Y', 'N'])
    if exercise_angina_val:
        parsed['ExerciseAngina'] = exercise_angina_val.upper()

    st_slope_val = parse_choice(data.get('st_slope'), 'ST Slope', ['Up', 'Flat', 'Down'])
    if st_slope_val:
        parsed['ST_Slope'] = st_slope_val.capitalize()

    return parsed, errors

@app.route('/')
def home():
    """Render main web application interface."""
    return render_template('index.html', model_ready=(model is not None))

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "columns_loaded": expected_columns is not None,
        "error": model_load_error
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Process input features and generate heart disease prediction."""
    if model is None or scaler is None or expected_columns is None:
        if not load_ml_artifacts():
            error_msg = "ML Model files are not properly loaded on server."
            if request.is_json:
                return jsonify({"success": False, "error": error_msg}), 500
            return render_template('index.html', error=error_msg, model_ready=False), 500

    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    parsed, errors = validate_and_parse_input(data)

    if errors:
        error_message = "Please enter valid information in all required fields."
        if request.is_json:
            return jsonify({"success": False, "errors": errors, "error": error_message}), 400
        return render_template('index.html', error=error_message, errors=errors, form_data=data, model_ready=True), 400

    try:
        # Construct feature dictionary matching expected dummy columns
        raw_input = {
            'Age': parsed['Age'],
            'RestingBP': parsed['RestingBP'],
            'Cholesterol': parsed['Cholesterol'],
            'FastingBS': parsed['FastingBS'],
            'MaxHR': parsed['MaxHR'],
            'Oldpeak': parsed['Oldpeak'],
            'Sex_' + parsed['Sex']: 1,
            'ChestPainType_' + parsed['ChestPainType']: 1,
            'RestingECG_' + parsed['RestingECG']: 1,
            'ExerciseAngina_' + parsed['ExerciseAngina']: 1,
            'ST_Slope_' + parsed['ST_Slope']: 1
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)

        # Classification (0 = No Risk Detected, 1 = Risk Detected)
        prediction_class = int(model.predict(scaled_input)[0])
        
        # Calculate model prediction probability score
        risk_probability = None
        if hasattr(model, "predict_proba"):
            risk_probability = round(float(model.predict_proba(scaled_input)[0][1]) * 100, 2)

        result_text = "Heart Disease Risk Detected" if prediction_class == 1 else "No Heart Disease Risk Detected"

        response_data = {
            "success": True,
            "prediction": prediction_class,
            "result_text": result_text,
            "risk_probability": risk_probability,
            "message": f"Prediction complete: {result_text} ({risk_probability}% model prediction probability)."
        }

        if request.is_json:
            return jsonify(response_data)

        return render_template(
            'index.html',
            prediction=prediction_class,
            result_text=result_text,
            risk_probability=risk_probability,
            form_data=data,
            model_ready=True
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        error_msg = "An internal processing error occurred. Please check input values and try again."
        if request.is_json:
            return jsonify({"success": False, "error": error_msg}), 500
        return render_template('index.html', error=error_msg, form_data=data, model_ready=True), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1']
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

