import os
import requests
import pandas as pd

def download_data(url: str, output_path: str):
    """Downloads a CSV file from a URL and saves it locally."""
    print(f"Downloading data from {url}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Data successfully saved to {output_path}")
    else:
        raise Exception(f"Failed to download data. Status code: {response.status_code}")

def load_data(file_path: str) -> pd.DataFrame:
    """Loads CSV data into a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

if __name__ == "__main__":
    DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Customer-Churn.csv"
    OUTPUT_FILE = os.path.join("data", "raw", "customer_churn.csv")
    download_data(DATA_URL, OUTPUT_FILE)
    df = load_data(OUTPUT_FILE)
    print(f"Loaded DataFrame with shape: {df.shape}")
