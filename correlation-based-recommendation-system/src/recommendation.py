# recommendation.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class RecommendationSystem:
    def __init__(self, processed_data):
        self.data = processed_data
        self.scaler = StandardScaler()
        self.similarity_matrix = None

    def preprocess_data(self):
        # Assuming the processed data has relevant features for recommendations
        features = self.data.select_dtypes(include=['float64', 'int64'])
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features

    def compute_similarity(self):
        scaled_data = self.preprocess_data()
        self.similarity_matrix = cosine_similarity(scaled_data)

    def get_recommendations(self, user_index, num_recommendations=5):
        if self.similarity_matrix is None:
            self.compute_similarity()
        
        # Get the similarity scores for the user
        user_scores = self.similarity_matrix[user_index]
        
        # Get the indices of the most similar items
        similar_indices = user_scores.argsort()[-num_recommendations-1:-1][::-1]
        
        # Return the recommended items
        return self.data.iloc[similar_indices]

def main():
    # Example usage
    processed_data = pd.read_csv('data/processed/processed_dataset.csv')
    recommender = RecommendationSystem(processed_data)
    recommendations = recommender.get_recommendations(user_index=0, num_recommendations=5)
    print(recommendations)

if __name__ == "__main__":
    main()