# utils.py - Updated for Railway
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

def get_temp_dir():
    """Get appropriate temp directory for Railway"""
    # Railway biasanya menggunakan /tmp, tapi fallback ke current dir
    temp_dirs = ["/tmp", "./temp", "."]
    for temp_dir in temp_dirs:
        try:
            if os.path.exists(temp_dir) and os.access(temp_dir, os.W_OK):
                return temp_dir
        except:
            continue
    return "."

def load_or_generate_similarity_matrix():
    """Load similarity matrix jika ada, atau generate jika tidak ada"""
    temp_dir = get_temp_dir()
    matrix_path = os.path.join(temp_dir, "cosine_similarity_matrix.npy")
    
    if os.path.exists(matrix_path):
        try:
            logger.info(f"Loading existing similarity matrix from {matrix_path}...")
            return np.load(matrix_path)
        except Exception as e:
            logger.warning(f"Failed to load existing matrix: {e}")
    
    logger.info("Generating new similarity matrix...")
    similarity_matrix = generate_similarity_matrix()
    
    # Save untuk next request
    try:
        logger.info(f"Saving similarity matrix to {matrix_path}...")
        np.save(matrix_path, similarity_matrix)
        logger.info("Similarity matrix saved successfully")
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
        logger.info(f"Loading models from: {MODELS_DIR}")
        
        # Check if models directory exists
        if not os.path.exists(MODELS_DIR):
            raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")
        
        logger.info("Loading TF-IDF vectorizer...")
        vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"TF-IDF vectorizer not found: {vectorizer_path}")
        tfidf_vectorizer = joblib.load(vectorizer_path)
        
        logger.info("Loading movies data...")
        movies_path = os.path.join(MODELS_DIR, "movies_content.csv")
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Movies data not found: {movies_path}")
        movies_df = pd.read_csv(movies_path)
        
        logger.info("Loading movie ID mappings...")
        mappings_path = os.path.join(MODELS_DIR, "movie_id_mappings.json")
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Movie ID mappings not found: {mappings_path}")
        with open(mappings_path, "r") as f:
            index_to_movie_id = json.load(f)
        
        movie_id_to_index = {v: int(k) for k, v in index_to_movie_id.items()}
        
        logger.info(f"Base data loaded successfully - {len(movies_df)} movies")
        
    except Exception as e:
        logger.error(f"Error loading base data: {e}")
        raise

def initialize_models():
    """Initialize semua models dan data"""
    global cosine_sim_matrix
    
    try:
        logger.info("Initializing models...")
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
            raise ValueError("TF-IDF vectorizer not loaded. Call initialize_models() first.")
            
        text = " ".join(genres + favorites).lower()
        return tfidf_vectorizer.transform([text]).toarray()
    except Exception as e:
        logger.error(f"Error creating user vector: {e}")
        raise

def get_recommendations(user_vector: np.ndarray, top_n: int = 5):
    """Compute similarity and return top-N movie info."""
    try:
        if movies_df is None or tfidf_vectorizer is None:
            raise ValueError("Models not loaded. Call initialize_models() first.")
            
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
                "similarity": float(sims[idx]),
                "rank": len(recommendations) + 1
            })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise

def get_movie_recommendations_by_id(movie_id: int, top_n: int = 10):
    """Get movie recommendations berdasarkan movie ID"""
    try:
        if cosine_sim_matrix is None or movie_id_to_index is None or movies_df is None:
            raise ValueError("Models not loaded")
        
        if movie_id not in movie_id_to_index:
            return {"error": f"Movie ID {movie_id} not found"}
        
        movie_idx = movie_id_to_index[movie_id]
        sim_scores = list(enumerate(cosine_sim_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Exclude movie itu sendiri
        
        recommended_movies = []
        for idx, score in sim_scores:
            movie_info = {
                'movie_id': int(movies_df.iloc[idx]['movieId']),
                'title': movies_df.iloc[idx]['title'],
                'genres': movies_df.iloc[idx]['genres'],
                'similarity_score': float(score)
            }
            recommended_movies.append(movie_info)
        
        return {
            'movie_id': movie_id,
            'recommendations': recommended_movies
        }
        
    except Exception as e:
        logger.error(f"Error getting movie recommendations: {e}")
        return {"error": f"Error: {str(e)}"}