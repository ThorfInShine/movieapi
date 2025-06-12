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

# Global variables
tfidf_vectorizer = None
cosine_sim_matrix = None
movies_df = None
index_to_movie_id = None
movie_id_to_index = None

def load_or_generate_similarity_matrix():
    """Load similarity matrix jika ada, atau generate jika tidak ada"""
    # Di Vercel, simpan di /tmp directory
    matrix_path = "/tmp/cosine_similarity_matrix.npy"
    
    if os.path.exists(matrix_path):
        logger.info("Loading existing similarity matrix from /tmp...")
        return np.load(matrix_path)
    else:
        logger.info("Generating new similarity matrix...")
        similarity_matrix = generate_similarity_matrix()
        
        # Save ke /tmp untuk next request
        try:
            logger.info("Saving similarity matrix to /tmp...")
            np.save(matrix_path, similarity_matrix)
        except Exception as e:
            logger.warning(f"Could not save similarity matrix: {e}")
        
        return similarity_matrix

def generate_similarity_matrix():
    """Generate cosine similarity matrix dari data movies"""
    try:
        global tfidf_vectorizer, movies_df
        
        logger.info("Generating TF-IDF matrix...")
        movie_features = movies_df['title'] + " " + movies_df['genres']
        tfidf_matrix = tfidf_vectorizer.transform(movie_features)
        
        logger.info("Calculating cosine similarity...")
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        logger.info(f"Similarity matrix generated with shape: {similarity_matrix.shape}")
        return similarity_matrix
        
    except Exception as e:
        logger.error(f"Error generating similarity matrix: {e}")
        raise e

def load_base_data():
    """Load data dasar"""
    global tfidf_vectorizer, movies_df, index_to_movie_id, movie_id_to_index
    
    try:
        logger.info("Loading TF-IDF vectorizer...")
        tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        
        logger.info("Loading movies data...")
        movies_df = pd.read_csv(os.path.join(MODELS_DIR, "movies_content.csv"))
        
        logger.info("Loading movie ID mappings...")
        with open(os.path.join(MODELS_DIR, "movie_id_mappings.json"), "r") as f:
            index_to_movie_id = json.load(f)
        
        movie_id_to_index = {v: int(k) for k, v in index_to_movie_id.items()}
        
        logger.info("Base data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading base data: {e}")
        raise

def initialize_models():
    """Initialize semua models dan data"""
    global cosine_sim_matrix
    
    try:
        load_base_data()
        cosine_sim_matrix = load_or_generate_similarity_matrix()
        
        logger.info("All models initialized successfully")
        logger.info(f"Loaded {len(movies_df)} movies")
        logger.info(f"Similarity matrix shape: {cosine_sim_matrix.shape}")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def get_user_vector(genres: list[str], favorites: list[str]) -> np.ndarray:
    """Convert user input to TF-IDF vector."""
    try:
        if tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not loaded")
            
        text = " ".join(genres + favorites).lower()
        return tfidf_vectorizer.transform([text]).toarray()
    except Exception as e:
        logger.error(f"Error creating user vector: {e}")
        raise

def get_recommendations(user_vector: np.ndarray, top_n: int = 5):
    """Compute similarity and return top-N movie info."""
    try:
        if movies_df is None or tfidf_vectorizer is None:
            raise ValueError("Models not loaded")
            
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
                "genres_list": movie_data["genres"].split("|") if "|" in movie_data["genres"] else [movie_data["genres"]],
                "similarity": float(sims[idx]),
                "rank": len(recommendations) + 1
            })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise