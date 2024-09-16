# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling non-numeric columns:
    - Convert date columns to numeric (VehicleAge).
    - Encode categorical columns using Label Encoding or One-Hot Encoding.
    """
    # Convert 'VehicleIntroDate' to 'VehicleAge'
    if 'VehicleIntroDate' in df.columns:
        df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')
        current_year = datetime.now().year
        df['VehicleAge'] = current_year - df['VehicleIntroDate'].dt.year
        df = df.drop(columns=['VehicleIntroDate'])  # Drop the original date column
    
    # Identify categorical columns (non-numeric columns)
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Apply Label Encoding for binary categorical columns
    binary_columns = ['IsVATRegistered', 'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 
                      'AlarmImmobiliser', 'TrackingDevice', 'CapitalOutstanding']
    
    for col in binary_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Ensure data type is correct
    
    # Apply One-Hot Encoding for multi-class categorical columns
    df = pd.get_dummies(df, drop_first=True)
    
    # Drop any remaining non-numeric columns that cannot be processed
    df = df.select_dtypes(exclude=['object', 'datetime'])

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
    binary_columns = ['IsVATRegistered', 'NewVehicle', 'WrittenOff', 
                      'Rebuilt', 'Converted', 'AlarmImmobiliser', 
                      'TrackingDevice', 'CapitalOutstanding']
    
    # Label encode binary columns
    for col in binary_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(int))  # Ensure object type conversion

    # List of multi-class categorical columns (one-hot encoding)
    multi_class_columns = ['Citizenship', 'LegalType', 'Title', 'Language', 
                           'Bank', 'AccountType', 'MaritalStatus', 'Gender', 
                           'Country', 'Province', 'MainCrestaZone', 'SubCrestaZone', 
                           'ItemType', 'VehicleType', 'make', 'Model', 'bodytype', 
                           'TermFrequency', 'ExcessSelected', 'CoverCategory', 
                           'CoverType', 'CoverGroup', 'Section', 'Product', 
                           'StatutoryClass', 'StatutoryRiskType']

    # One-hot encode the multi-class categorical columns
    df = pd.get_dummies(df, columns=multi_class_columns, drop_first=True)
    
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
