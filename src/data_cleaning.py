import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the dataset:
    - Removes duplicates
    - Handles missing values
    - Caps outliers using IQR method
    """
    print("\nStarting Data Cleaning...")
    
    # 1. Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows.")

    # 2. Handle Missing Values
    # Age: Fill with median
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Cabin: Fill with 'U' for Unknown
    if 'Cabin' in df.columns:
        df['Cabin'] = df['Cabin'].fillna('U')
        
    # Embarked: Fill with mode
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 3. Handle Outliers in 'Fare' using IQR
    if 'Fare' in df.columns:
        Q1 = df['Fare'].quantile(0.25)
        Q3 = df['Fare'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outilers
        df['Fare'] = np.where(df['Fare'] > upper_bound, upper_bound, df['Fare'])
        df['Fare'] = np.where(df['Fare'] < lower_bound, lower_bound, df['Fare'])
        print(f"Capped outliers in Fare column using IQR boundaries: [{lower_bound:.2f}, {upper_bound:.2f}]")

    print("Data Cleaning Complete.\n")
    return df
