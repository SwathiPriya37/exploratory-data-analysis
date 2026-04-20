import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def feature_engineering(df):
    """
    Performs Feature Engineering:
    - Creates 'FamilySize' from 'SibSp' and 'Parch'
    - Creates 'IsAlone' boolean feature
    - Drops unnecessary string/id columns
    - Encodes categoricals (LabelEncoding & One-Hot Encoding)
    - Standardizes numerical variables
    """
    print("Starting Feature Engineering...")
    
    # Create new feature 1: FamilySize
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Create new feature 2: IsAlone (1 if traveling alone, 0 otherwise)
        df['IsAlone'] = 1
        df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
        print("Generated 'FamilySize' and 'IsAlone' features.")

    # Drop unneeded columns
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Encode Categorical Variables
    # 1. Label Encoding for binary or ordinal like 'Sex'
    if 'Sex' in df_clean.columns:
        le = LabelEncoder()
        df_clean['Sex'] = le.fit_transform(df_clean['Sex'])
        print("Label Encoded 'Sex'.")

    # 2. One-Hot Encoding for Nominal categories like 'Embarked'
    if 'Embarked' in df_clean.columns:
        df_engine = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True)
        print("One-Hot Encoded 'Embarked'.")
    else:
        df_engine = df_clean.copy()

    # Standardization
    if 'Survived' in df_engine.columns:
        X = df_engine.drop('Survived', axis=1)
        y = df_engine['Survived']
        
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        print("Standardized Numerical Features.")
    else:
        # If target doesn't exist (like a pure prediction pipeline), scale all
        scaler = StandardScaler()
        final_df = pd.DataFrame(scaler.fit_transform(df_engine), columns=df_engine.columns)
        
    print("Feature Engineering Complete.\n")
    return final_df
