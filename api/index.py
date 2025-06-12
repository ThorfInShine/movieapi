# api/index.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path untuk import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import initialize_models, get_user_vector, get_recommendations, movies_df
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Fallback imports
    initialize_models = None
    get_user_vector = None
    get_recommendations = None
    movies_df = None

app = FastAPI(
    title="Movie Recommendation API",
    description="API untuk mendapatkan rekomendasi film",
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

# Global flag untuk track initialization
models_initialized = False

@app.on_event("startup")
async def startup_event():
    """Load semua data saat aplikasi start"""
    global models_initialized
    try:
        if initialize_models:
            logger.info("Starting application...")
            initialize_models()
            models_initialized = True
            logger.info("Application started successfully!")
        else:
            logger.error("initialize_models function not available")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        models_initialized = False

@app.get("/")
async def root():
    return {
        "message": "Movie Recommendation API",
        "status": "running" if models_initialized else "error",
        "models_loaded": models_initialized,
        "endpoints": {
            "recommendations": "/recommend",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/recommend")
async def recommend_movies(request: RecommendationRequest):
    """Get rekomendasi berdasarkan user preferences"""
    if not models_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Models are still loading or failed to load, please try again"
        )
    
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
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if models_initialized else "unhealthy",
        "models_loaded": models_initialized,
        "total_movies": len(movies_df) if movies_df is not None else 0
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint untuk debugging"""
    try:
        if not models_initialized:
            return {"status": "Models not initialized"}
        
        # Test dengan data sample
        test_genres = ["Action", "Adventure"]
        test_favorites = ["Spider-Man"]
        
        user_vec = get_user_vector(test_genres, test_favorites)
        recommendations = get_recommendations(user_vec, 3)
        
        return {
            "test": "success",
            "sample_input": {
                "genres": test_genres,
                "favorites": test_favorites
            },
            "sample_recommendations": recommendations
        }
    except Exception as e:
        return {
            "test": "failed",
            "error": str(e)
        }

# Handler untuk Vercel
handler = app