from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def perform_semantic_analysis(data_path):
    # Load the processed dataset
    processed_data = pd.read_csv(data_path)

    # Assuming the dataset has a 'description' column for semantic analysis
    descriptions = processed_data['description'].fillna('')

    # Create TF-IDF vectors for the descriptions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Calculate cosine similarity between the items
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim

# Example usage
if __name__ == "__main__":
    data_path = '../data/processed/processed_dataset.csv'
    similarity_matrix = perform_semantic_analysis(data_path)
    print(similarity_matrix)