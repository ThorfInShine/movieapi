# utils.py
import numpy as np
import pandas as pd
import json
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# Get the correct path for models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    # Load assets with absolute paths
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    cosine_sim_matrix = np.load(os.path.join(MODELS_DIR, "cosine_similarity_matrix.npy"))
    movies_df = pd.read_csv(os.path.join(MODELS_DIR, "movies_content.csv"))
    
    with open(os.path.join(MODELS_DIR, "movie_id_mappings.json"), "r") as f:
        index_to_movie_id = json.load(f)
    
    movie_id_to_index = {v: int(k) for k, v in index_to_movie_id.items()}
    
    logger.info("All models loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

def get_user_vector(genres: list[str], favorites: list[str]) -> np.ndarray:
    """Convert user input to TF-IDF vector."""
    try:
        text = " ".join(genres + favorites).lower()
        return tfidf_vectorizer.transform([text]).toarray()
    except Exception as e:
        logger.error(f"Error creating user vector: {e}")
        raise

def get_recommendations(user_vector: np.ndarray, top_n: int = 5):
    """Compute similarity and return top-N movie info."""
    try:
        movie_features = movies_df['title'] + " " + movies_df['genres']
        movie_vectors = tfidf_vectorizer.transform(movie_features)
        
        sims = cosine_similarity(user_vector, movie_vectors)[0]
        top_indices = sims.argsort()[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            movie_data = movies_df.iloc[idx]
            recommendations.append({
                "movieId": int(movie_data["movieId"]),
                "title": movie_data["title"],
                "genres": movie_data["genres"],
                "genres_list": movie_data["genres"].split("|"),
                "similarity": float(sims[idx]),
                "rank": len(recommendations) + 1
            })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise