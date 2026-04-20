import pandas as pd

def load_data(filepath='../data/titanic.csv'):
    """
    Load dataset from the given filepath.
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df
