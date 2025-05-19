# src/preprocess.py

import pandas as pd

def load_customer_data(file_path: str) -> pd.DataFrame:
    """
    Loads customer profile data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        raise

def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans customer data:
    - Standardizes column names
    - Fills missing values
    - Trims whitespace from string fields
    """
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Fill missing values
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Unknown").str.strip()
    for col in df.select_dtypes(include=['int', 'float']).columns:
        df[col] = df[col].fillna(df[col].median())

    print(f"✅ Cleaned data with shape: {df.shape}")
    return df
