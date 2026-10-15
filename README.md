# ❤️ Heart Disease Prediction Web Application

An end-to-end Machine Learning web application designed to predict heart disease risk based on user-provided health parameters. Built with **Python**, **Flask**, and **Scikit-Learn**, this application provides a responsive interface for real-time model prediction.

---

## 📌 About the Project

This project uses a pre-trained **Logistic Regression** classifier to process 11 user-facing health parameters, transform them into a 15-column dummy-encoded feature representation, apply feature scaling via `StandardScaler`, and return a binary risk classification alongside a model prediction probability score.

---

## ✨ Key Features

- **11 User-Facing Health Parameters**: Input form supporting 11 health inputs with client-side and server-side range validation.
- **Automated Feature Preprocessing**: Dummy encoding for categorical variables aligned with training schema (`columns.pkl`).
- **Feature Scaling**: Normalization using pre-fitted `StandardScaler` (`scaler.pkl`).
- **Logistic Regression Inference**: Binary classification output (Risk Detected / No Risk Detected) using `Logistic_Regression.pkl`.
- **Model Prediction Probability**: Calculates score using `model.predict_proba()` when supported by the classifier.
- **RESTful API Endpoint**: Dedicated `/predict` endpoint supporting JSON and HTML form submissions.
- **Automated Unit Testing**: Test suite executed via `pytest`.

---

## 📊 11 User-Facing Parameters & 15 Model Features

### 11 User-Facing Input Parameters
1. **Age**: Patient age in years (18 – 120)
2. **Sex**: Biological sex (`M` = Male, `F` = Female)
3. **Chest Pain Type**: Chest pain category (`ATA` = Atypical Angina, `NAP` = Non-Anginal Pain, `TA` = Typical Angina, `ASY` = Asymptomatic)
4. **Resting Blood Pressure**: Resting BP on admission in mm Hg (60 – 240)
5. **Serum Cholesterol**: Serum cholesterol in mg/dl (0 – 600)
6. **Fasting Blood Sugar**: Fasting BS > 120 mg/dl (`1` = True, `0` = False)
7. **Resting ECG**: Resting electrocardiographic results (`Normal`, `ST` = ST-T Wave Abnormality, `LVH` = Left Ventricular Hypertrophy)
8. **Maximum Heart Rate**: Maximum heart rate achieved in bpm (40 – 220)
9. **Exercise Induced Angina**: Angina induced by exercise (`Y` = Yes, `N` = No)
10. **Oldpeak**: ST depression induced by exercise relative to rest (-2.6 – 10.0)
11. **ST Slope**: Slope of the peak exercise ST segment (`Up` = Upsloping, `Flat` = Flat, `Down` = Downsloping)

### 15 Processed Model Features (`columns.pkl`)
The backend transforms the 11 user inputs into the exact 15-column feature order required by the model:
1. `Age`
2. `RestingBP`
3. `Cholesterol`
4. `FastingBS`
5. `MaxHR`
6. `Oldpeak`
7. `Sex_M`
8. `ChestPainType_ATA`
9. `ChestPainType_NAP`
10. `ChestPainType_TA`
11. `RestingECG_Normal`
12. `RestingECG_ST`
13. `ExerciseAngina_Y`
14. `ST_Slope_Flat`
15. `ST_Slope_Up`

---

## 🔬 Machine Learning Pipeline

```text
User Inputs (11 parameters)
   ↓
Input Parsing & Boundary Validation
   ↓
One-Hot Dummy Encoding (15 columns aligned with columns.pkl)
   ↓
StandardScaler Preprocessing (scaler.pkl)
   ↓
Logistic Regression Inference (Logistic_Regression.pkl)
   ↓
Model Prediction Probability (model.predict_proba())
   ↓
Output Rendering (Web UI / JSON)
```

---

## 💻 Tech Stack & Dependencies

- **Backend**: Python 3.10+, Flask
- **Machine Learning**: Scikit-Learn, Pandas, Joblib
- **Testing**: Pytest
- **Production Server**: Gunicorn

---

## 📁 Project Structure

```text
heartdisease_predicton/
│
├── app.py                      # Flask application routes and prediction logic
├── requirements.txt            # Pinned dependency requirements
├── runtime.txt                 # Deployment runtime specifier
├── README.md                   # Repository documentation
├── .gitignore                  # Git ignore rules
│
├── models/                     # Pre-trained ML Artifacts
│   ├── Logistic_Regression.pkl # Trained Logistic Regression model
│   ├── scaler.pkl              # Fitted StandardScaler instance
│   └── columns.pkl             # Model feature column definition (15 columns)
│
├── templates/                  # HTML Templates
│   └── index.html              # Main application web interface
│
├── static/                     # Web Assets
│   ├── css/
│   │   └── style.css           # Glassmorphism UI stylesheet
│   └── js/
│       └── script.js           # DOM handling and AJAX submission script
│
└── tests/                      # Unit Test Suite
    └── test_app.py             # Route and validation tests
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/saksham196/heartdisease_predicton.git
cd heartdisease_predicton
```

### 2. Create & Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Start the Flask application server:

```bash
python app.py
```

Access the application in your browser at:
```text
http://127.0.0.1:5000
```

---

## 🧪 Running Unit Tests

Run the automated test suite using `pytest`:

```bash
python -m pytest
```

---

## ⚠️ Medical Disclaimer

> **Disclaimer:** This application is built for educational and demonstration purposes only. It does not provide medical advice, diagnosis, or clinical validation. Always consult a qualified healthcare professional for medical concerns.

---

## 🔗 Repository

- **GitHub Repository**: [https://github.com/saksham196/heartdisease_predicton](https://github.com/saksham196/heartdisease_predicton)
