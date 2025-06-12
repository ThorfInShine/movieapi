# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import (
    initialize_models, 
    get_user_vector, 
    get_recommendations,
    get_movie_recommendations_by_id,
    movies_df
)

app = FastAPI(
    title="Movie Recommendation API",
    description="API untuk mendapatkan rekomendasi film berdasarkan user preferences dan movie similarity",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    genres: List[str]
    favorites: List[str]
    top_n: int = 5

@app.on_event("startup")
async def startup_event():
    """
    Load semua data saat aplikasi start
    """
    try:
        logger.info("Starting application...")
        initialize_models()
        logger.info("Application started successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise e

@app.get("/")
async def root():
    return {
        "message": "Movie Recommendation API",
        "status": "running",
        "endpoints": {
            "user_recommendations": "/recommend",
            "movie_recommendations": "/movies/{movie_id}/recommendations",
            "health": "/health"
        },
        "total_movies": len(movies_df) if movies_df is not None else 0
    }

@app.post("/recommend")
async def recommend_movies(request: RecommendationRequest):
    """
    Get rekomendasi berdasarkan user preferences (genres dan favorites)
    """
    try:
        user_vec = get_user_vector(request.genres, request.favorites)
        top_movies = get_recommendations(user_vec, request.top_n)
        
        return {
            "user_input": {
                "genres": request.genres,
                "favorites": request.favorites,
                "top_n": request.top_n
            },
            "recommendations": top_movies
        }
        
    except Exception as e:
        logger.error(f"Error in recommend_movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies/{movie_id}/recommendations")
async def get_similar_movies(movie_id: int, top_n: int = 10):
    """
    Get rekomendasi berdasarkan movie ID (similar movies)
    """
    try:
        if top_n > 50:
            top_n = 50  # Limit maksimal 50 recommendations
        
        recommendations = get_movie_recommendations_by_id(movie_id, top_n)
        
        if "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_similar_movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": movies_df is not None,
        "total_movies": len(movies_df) if movies_df is not None else 0
    }

# Test endpoint untuk debugging
@app.get("/test")
async def test_endpoint():
    """Test endpoint untuk memastikan semua berjalan"""
    try:
        # Test user recommendation
        test_genres = ["Action", "Adventure"]
        test_favorites = ["Spider-Man", "Batman"]
        
        user_vec = get_user_vector(test_genres, test_favorites)
        recommendations = get_recommendations(user_vec, 3)
        
        return {
            "test": "success",
            "sample_recommendations": recommendations
        }
        
    except Exception as e:
        return {
            "test": "failed",
            "error": str(e)
        }