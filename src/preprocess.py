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
    df = df.copy()

    # Drop obviously irrelevant columns (like ID columns, if any)
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df.drop(columns=id_cols, inplace=True, errors='ignore')

    # Drop columns with too many missing values
    df = df.dropna(thresh=len(df) * 0.5, axis=1)

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Handle categorical variables
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"✅ Cleaned data with shape: {df.shape}")
    return df
