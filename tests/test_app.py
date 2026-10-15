import os
import sys
import unittest
import json

# Add parent directory to path so app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, MODELS_DIR

class HeartDiseaseAppTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test client before each test."""
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_required_model_files_exist(self):
        """Test that all required ML model artifact files exist in models/."""
        required_files = ["Logistic_Regression.pkl", "scaler.pkl", "columns.pkl"]
        for filename in required_files:
            file_path = os.path.join(MODELS_DIR, filename)
            self.assertTrue(
                os.path.exists(file_path),
                f"Missing required model artifact file: {file_path}"
            )

    def test_home_page_loads(self):
        """Test homepage route GET / returns HTTP 200."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Heart Disease Prediction', response.data)

    def test_health_endpoint(self):
        """Test health check route GET /health returns HTTP 200 and healthy status."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])
        self.assertTrue(data['scaler_loaded'])
        self.assertTrue(data['columns_loaded'])

    def test_valid_prediction_json(self):
        """Test valid patient input data POST /predict returns prediction classification."""
        valid_payload = {
            "age": 45,
            "sex": "M",
            "chest_pain": "ATA",
            "resting_bp": 120,
            "cholesterol": 210,
            "fasting_bs": 0,
            "resting_ecg": "Normal",
            "max_hr": 160,
            "exercise_angina": "N",
            "oldpeak": 0.5,
            "st_slope": "Up"
        }
        response = self.client.post(
            '/predict',
            data=json.dumps(valid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn(data['prediction'], [0, 1])
        self.assertIn('result_text', data)
        self.assertIsNotNone(data['risk_probability'])

    def test_invalid_input_out_of_bounds(self):
        """Test invalid out-of-bounds input returns HTTP 400 validation error."""
        invalid_payload = {
            "age": 5, # Invalid age < 18
            "sex": "M",
            "chest_pain": "ATA",
            "resting_bp": 120,
            "cholesterol": 210,
            "fasting_bs": 0,
            "resting_ecg": "Normal",
            "max_hr": 160,
            "exercise_angina": "N",
            "oldpeak": 0.5,
            "st_slope": "Up"
        }
        response = self.client.post(
            '/predict',
            data=json.dumps(invalid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('errors', data)

    def test_invalid_categorical_input(self):
        """Test invalid categorical value returns HTTP 400 validation error."""
        invalid_payload = {
            "age": 45,
            "sex": "INVALID_SEX",
            "chest_pain": "ATA",
            "resting_bp": 120,
            "cholesterol": 210,
            "fasting_bs": 0,
            "resting_ecg": "Normal",
            "max_hr": 160,
            "exercise_angina": "N",
            "oldpeak": 0.5,
            "st_slope": "Up"
        }
        response = self.client.post(
            '/predict',
            data=json.dumps(invalid_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])

if __name__ == '__main__':
    unittest.main()
