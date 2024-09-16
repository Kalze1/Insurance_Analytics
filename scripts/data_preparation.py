# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer






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

# Encode Categorical Data
def encode_categorical_data(df: pd.DataFrame):
    """
    Encodes categorical data using One-Hot Encoding for multi-class features and Label Encoding for binary features.
    """
    # Use Label Encoding for binary categorical features
    binary_columns = ['IsVATRegistered', 'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted']
    for col in binary_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Use One-Hot Encoding for other categorical features
    df = pd.get_dummies(df, columns=['Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 
                                     'AccountType', 'MaritalStatus', 'Gender', 'Country', 
                                     'Province', 'MainCrestaZone', 'SubCrestaZone', 
                                     'ItemType', 'VehicleType', 'make', 'Model', 'bodytype'], drop_first=True)
    
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
