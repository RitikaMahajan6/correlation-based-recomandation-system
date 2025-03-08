# data_preprocessing.py

import pandas as pd

def load_and_process_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Basic data cleaning
    df.dropna(inplace=True)  # Remove missing values
    df.drop_duplicates(inplace=True)  # Remove duplicate entries

    # Example transformation: Convert categorical variables to numerical
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category').cat.codes

    # Save the processed dataset
    processed_file_path = 'data/processed/processed_dataset.csv'
    df.to_csv(processed_file_path, index=False)

    return df