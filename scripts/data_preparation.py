# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime

import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def calculate_vehicle_age(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    """
    Adds a 'VehicleAge' column based on the 'RegistrationYear' column.
    
    Args:
    df (pd.DataFrame): Input DataFrame containing 'RegistrationYear'.
    current_year (int): Optionally specify the current year. Defaults to the current year.

    Returns:
    pd.DataFrame: Updated DataFrame with 'VehicleAge' column.
    """
    if current_year is None:
        current_year = datetime.now().year  # Default to the current year

    # Calculate vehicle age based on 'RegistrationYear'
    df['VehicleAge'] = current_year - df['RegistrationYear']
    
    return df


# Feature Engineering
def feature_engineering(df: pd.DataFrame):
    """
    Adds new features to the dataset that could be relevant to 'TotalPremium' and 'TotalClaims'.
    """
    # Example feature: Age of the vehicle based on RegistrationYear
    df['VehicleAge'] = 2024 - df['RegistrationYear']
    
    # Example feature: Estimate risk based on VehicleType and SumInsured
    df['RiskFactor'] = df['SumInsured'] / df['VehicleAge']
    
    return df




def encode_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical data using One-Hot Encoding for multi-class features
    and Label Encoding for binary features.
    """

    # List of binary categorical columns (label encoding)
    binary_columns = ['NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 
                      'AlarmImmobiliser', 'TrackingDevice', 'CapitalOutstanding']

    # Handle potential non-numeric string values in binary columns
    for col in binary_columns:
        if df[col].dtype == 'object':
            # Check if there are any non-numeric values and handle them
            if df[col].str.isnumeric().all():
                df[col] = df[col].astype(int)  # Convert if all values are numeric
            else:
                # Example: map non-numeric values to integers (custom mapping may be required)
                df[col] = df[col].map({
                    'Yes': 1, 'No': 0, 'More than 6 months': 2, 'Less than 6 months': 1,  # Adjust this mapping based on your data
                    'None': 0
                }).fillna(0)  # Fallback for unmapped values

    # Label encode binary columns
    for col in binary_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(int))  # Ensure type conversion if necessary

    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Apply one-hot encoding to categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    
    return df


# Train-Test Split
def train_test_splitting(df: pd.DataFrame, target_cols: list, test_size: float = 0.3):
    """
    Splits the data into train and test sets, with a given test size.
    """
    X = df.drop(columns=target_cols)  # Features
    y = df[target_cols]  # Targets: TotalPremium and TotalClaims
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

#  Main Function
def prepar_data(df):
    # Feature Engineering
    df = feature_engineering(df)
    
    # Encode Categorical Data
    df = encode_categorical_data(df)
    
    # Split the data
    target_cols = ['TotalPremium', 'TotalClaims']
    X_train, X_test, y_train, y_test = train_test_splitting(df, target_cols)
    
    return X_train, X_test, y_train, y_test
