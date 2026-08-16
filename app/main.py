import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__)

# Load model pipeline
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "customer_churn_pipeline.joblib"))
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
        else:
            print(f"Warning: Model not found at {MODEL_PATH}. Prediction functionality will fail until trained.")
    return pipeline

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    clf_pipeline = get_pipeline()
    if clf_pipeline is None:
        return jsonify({"error": "Model not trained yet. Please run train.py first."}), 500
        
    try:
        # Get data from JSON or form parameters
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
            
        # Parse numeric inputs
        numeric_fields = {
            'SeniorCitizen': int,
            'tenure': int,
            'MonthlyCharges': float,
            'TotalCharges': float
        }
        
        for field, func in numeric_fields.items():
            if field in data and data[field] != '':
                data[field] = func(data[field])
            else:
                data[field] = 0
                
        # Handle TotalCharges if empty or space
        if 'TotalCharges' not in data or data['TotalCharges'] == 0:
            # Estimate total charges if missing
            data['TotalCharges'] = data['tenure'] * data['MonthlyCharges']

        # Ensure correct column ordering for scikit-learn pipeline
        expected_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
        ]
        
        # Fill standard values for any missing columns
        for col in expected_cols:
            if col not in data:
                if col in ['gender']:
                    data[col] = 'Male'
                elif col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                    data[col] = 'No'
                elif col in ['MultipleLines']:
                    data[col] = 'No'
                elif col in ['InternetService']:
                    data[col] = 'DSL'
                elif col in ['Contract']:
                    data[col] = 'Month-to-month'
                elif col in ['PaymentMethod']:
                    data[col] = 'Electronic check'
                else:
                    data[col] = 'No'
                    
        # Make DataFrame
        df_input = pd.DataFrame([data])[expected_cols]
        
        # Predict probability
        prob = clf_pipeline.predict_proba(df_input)[0][1]
        prediction = int(clf_pipeline.predict(df_input)[0])
        
        # Determine risk level
        risk_level = "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low"
        
        # Provide recommendations based on characteristics
        recommendations = []
        if prediction == 1:
            if data['Contract'] == 'Month-to-month':
                recommendations.append("Offer a discounted 1-year or 2-year contract transition.")
            if data['InternetService'] == 'Fiber optic':
                recommendations.append("Fiber optic users show higher churn. Investigate technical issues or review pricing.")
            if data['OnlineSecurity'] == 'No':
                recommendations.append("Propose adding Online Security add-on at a discounted rate to build retention.")
            if data['TechSupport'] == 'No':
                recommendations.append("Suggest scheduling a proactive customer success callback or adding Tech Support services.")
            if data['tenure'] < 12:
                recommendations.append("New customer (tenure < 1 year). Target with welcome/loyalty incentives.")
            if not recommendations:
                recommendations.append("Offer a standard customer appreciation coupon or review service feedback.")
        else:
            recommendations.append("Maintain current services. Monitor usage pattern changes.")
            
        return jsonify({
            "status": "success",
            "churn_probability": float(prob),
            "will_churn": bool(prediction),
            "risk_level": risk_level,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    # In production, Flask should be run on a web server like Gunicorn/Waitress.
    # For local testing:
    app.run(debug=True, host="0.0.0.0", port=5000)
