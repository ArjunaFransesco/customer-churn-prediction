import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from data_loader import download_data, load_data

def preprocess_and_split(df: pd.DataFrame):
    """Preprocesses raw customer churn data and splits it into train/test sets."""
    print("Preprocessing data...")
    df = df.copy()
    
    # 1. Handle missing values in TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    # Fill with median
    total_charges_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)
    
    # 2. Separate features and target
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Identify column types
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in X.columns if col not in numerical_cols]
    
    # 3. Create Preprocessing Pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # 4. Train-Test Split with Stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, preprocessor

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluates the model and prints classification report and ROC-AUC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n================ {model_name} Evaluation ================")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Create evaluation plot directory
    os.makedirs("reports/figures", exist_ok=True)
    
    # Plot ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', color='darkorange', lw=2)
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ax2.plot(recall, precision, label=f'{model_name}', color='blue', lw=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(f"reports/figures/{model_name.lower().replace(' ', '_')}_evaluation.png")
    plt.close()
    
    return roc_auc

def main():
    DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    RAW_PATH = os.path.join("data", "raw", "customer_churn.csv")
    
    # Download data if not exists
    if not os.path.exists(RAW_PATH):
        download_data(DATA_URL, RAW_PATH)
        
    df = load_data(RAW_PATH)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    
    # Models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), random_state=42, eval_metric='logloss')
    }
    
    best_score = 0
    best_model_name = ""
    best_pipeline = None
    
    for name, clf in models.items():
        # Build training pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        score = evaluate_model(pipeline, X_test, y_test, model_name=name)
        
        if score > best_score:
            best_score = score
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\nBest Model: {best_model_name} with ROC-AUC {best_score:.4f}")
    
    # Hyperparameter tuning on the best model (using Random Forest or XGBoost)
    print(f"Hyperparameter tuning on {best_model_name}...")
    if best_model_name == "Random Forest":
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 15, None],
            'classifier__min_samples_split': [2, 5, 10]
        }
    elif best_model_name == "XGBoost":
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }
    else: # Logistic Regression
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0]
        }
        
    grid_search = GridSearchCV(best_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    tuned_pipeline = grid_search.best_estimator_
    tuned_score = evaluate_model(tuned_pipeline, X_test, y_test, model_name=f"Tuned {best_model_name}")
    print(f"Tuned Model Parameter: {grid_search.best_params_}")
    
    # Save the best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(tuned_pipeline, "models/customer_churn_pipeline.joblib")
    print("Best pipeline saved successfully to models/customer_churn_pipeline.joblib")

if __name__ == "__main__":
    main()
