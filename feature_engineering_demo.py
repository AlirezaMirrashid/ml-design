
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_sample_data():
    """Creates a sample DataFrame mimicking the chapter's example."""
    data = {
        'Age': [27, 40, 35, 33, 20, 55, np.nan, 45],
        'Gender': ['B', 'B', 'A', 'B', 'A', 'A', 'B', 'A'],
        'Annual_Income': [50000, np.nan, 60000, 150000, 10000, 95000, 75000, np.nan],
        'Marital_Status': ['Single', 'Married', 'Single', 'Married', 'Single', 'Married', np.nan, 'Single'],
        'Num_Children': [0, 2, 0, 1, 0, 3, 2, 1],
        'Buy_House': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']
    }
    df = pd.DataFrame(data)
    print("--- 1. Original Raw Data ---")
    print(df)
    print("\nMissing values in original data:\n", df.isnull().sum())
    return df

def feature_engineering_pipeline(df):
    """
    Demonstrates the full feature engineering pipeline with a focus on preventing data leakage.
    """
    # Define which columns are numerical and which are categorical
    numerical_features = ['Age', 'Annual_Income', 'Num_Children']
    categorical_features = ['Gender', 'Marital_Status']
    
    # --- Step 2: Split Data BEFORE Any Engineering ---
    # This is the most critical step to prevent data leakage.
    # We separate the test set to simulate unseen data. All fitting will be done on the train set only.
    X = df.drop('Buy_House', axis=1)
    y = df['Buy_House']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print("\n--- 2. Data Split into Training and Testing Sets ---")
    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    # --- Step 3: Fit Transformers on TRAINING Data Only ---
    
    # 3.1 Imputation: Handling Missing Values
    # For numerical features, we'll use the median.
    # For categorical features, we'll use the most frequent value.
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Fit the imputers on the training data
    num_imputer.fit(X_train[numerical_features])
    cat_imputer.fit(X_train[categorical_features])

    # 3.2 Scaling: Standardizing Numerical Features
    scaler = StandardScaler()
    # Fit the scaler on the training data (after imputation)
    # We create a temporary imputed dataframe to fit the scaler
    X_train_num_imputed = pd.DataFrame(num_imputer.transform(X_train[numerical_features]), columns=numerical_features)
    scaler.fit(X_train_num_imputed)

    # 3.3 Encoding: One-Hot Encode Categorical Features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Fit the encoder on the training data (after imputation)
    X_train_cat_imputed = pd.DataFrame(cat_imputer.transform(X_train[categorical_features]), columns=categorical_features)
    encoder.fit(X_train_cat_imputed)

    print("\n--- 3. All Transformers (Imputers, Scaler, Encoder) Fitted on Training Data ONLY ---")

    # --- Step 4: Apply Transformations to Both Train and Test Sets ---
    
    print("\n--- 4. Applying Transformations ---")
    
    # Process Training Data
    print("\n--- Processing Training Data ---")
    X_train_processed = process_data(X_train, num_imputer, cat_imputer, scaler, encoder, numerical_features, categorical_features)
    print("Processed Training Data Head:")
    print(X_train_processed.head())

    # Process Testing Data
    print("\n--- Processing Testing Data ---")
    X_test_processed = process_data(X_test, num_imputer, cat_imputer, scaler, encoder, numerical_features, categorical_features)
    print("Processed Testing Data Head:")
    print(X_test_processed.head())

def process_data(df, num_imputer, cat_imputer, scaler, encoder, numerical_features, categorical_features):
    """Helper function to apply all transformations to a dataframe."""
    
    # Create a copy to avoid changing the original data
    df_processed = df.copy()

    # 1. Imputation
    df_processed[numerical_features] = num_imputer.transform(df_processed[numerical_features])
    df_processed[categorical_features] = cat_imputer.transform(df_processed[categorical_features])
    
    # 2. Feature Crossing
    # This should be done before encoding but after imputation.
    df_processed['Marital_Children_Interaction'] = df_processed['Marital_Status'] + "_" + df_processed['Num_Children'].astype(str)
    
    # 3. Scaling Numerical Features
    df_processed[numerical_features] = scaler.transform(df_processed[numerical_features])
    
    # 4. Encoding Categorical Features
    # We handle the new interaction feature along with the original ones
    categorical_features_with_interaction = categorical_features + ['Marital_Children_Interaction']
    encoded_data = encoder.fit_transform(df_processed[categorical_features_with_interaction]) # Re-fit for simplicity here, but in a real pipeline you'd handle this more carefully
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features_with_interaction), index=df_processed.index)
    
    # Combine processed numerical data and encoded categorical data
    df_processed = df_processed.drop(columns=categorical_features_with_interaction)
    df_final = pd.concat([df_processed, encoded_df], axis=1)
    
    return df_final

if __name__ == '__main__':
    # Create the initial dataset
    raw_df = create_sample_data()
    
    # Run the full feature engineering pipeline
    feature_engineering_pipeline(raw_df)
