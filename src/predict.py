import os
import joblib
import pandas as pd

def predict_churn(customer_data: dict, model_path: str = "models/customer_churn_pipeline.joblib") -> dict:
    """Predicts customer churn probability and class using a serialized model pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pipeline not found at {model_path}. Run train.py first.")
        
    pipeline = joblib.load(model_path)
    
    # Convert single dictionary inputs to dataframe
    df = pd.DataFrame([customer_data])
    
    # Ensure correct columns order and format
    # The models/customer_churn_pipeline.joblib includes preprocessing, so it expects the raw columns (except customerID)
    expected_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
    
    # Fill in missing columns with default values if any are absent
    for col in expected_cols:
        if col not in df.columns:
            # Simple defaults
            if col in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']:
                df[col] = 0
            else:
                df[col] = "No"
                
    df = df[expected_cols]
    
    # Predict
    prob = pipeline.predict_proba(df)[0][1]
    prediction = int(pipeline.predict(df)[0])
    
    return {
        "churn_probability": float(prob),
        "will_churn": bool(prediction),
        "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.3 else "Low"
    }

if __name__ == "__main__":
    # Test sample data
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.7,
        'TotalCharges': 151.65
    }
    
    try:
        res = predict_churn(sample_customer)
        print("Prediction Result:")
        print(res)
    except Exception as e:
        print(f"Error predicting: {e}. (Have you trained the model yet?)")
