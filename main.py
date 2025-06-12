# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    description="API untuk mendapatkan rekomendasi film berdasarkan ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
initialization_error = None

@app.on_event("startup")
async def startup_event():
    """Load semua data saat aplikasi start"""
    global models_initialized, initialization_error
    try:
        logger.info("Starting Movie Recommendation API...")
        initialize_models()
        models_initialized = True
        logger.info("✅ Application started successfully!")
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        models_initialized = False
        initialization_error = str(e)

@app.get("/")
async def root():
    """Root endpoint dengan informasi API"""
    return {
        "message": "🎬 Movie Recommendation API",
        "status": "running" if models_initialized else "error",
        "models_loaded": models_initialized,
        "total_movies": len(movies_df) if movies_df is not None else 0,
        "initialization_error": initialization_error,
        "endpoints": {
            "user_recommendations": "POST /recommend",
            "movie_recommendations": "GET /movies/{movie_id}/recommendations",
            "health_check": "GET /health",
            "api_docs": "GET /docs",
            "test": "GET /test"
        },
        "example_request": {
            "url": "/recommend",
            "method": "POST",
            "body": {
                "genres": ["Action", "Adventure"],
                "favorites": ["Spider-Man", "Batman"],
                "top_n": 5
            }
        }
    }

@app.post("/recommend")
async def recommend_movies(request: RecommendationRequest):
    """Get rekomendasi berdasarkan user preferences"""
    if not models_initialized:
        raise HTTPException(
            status_code=503, 
            detail=f"Models are not ready. Error: {initialization_error}"
        )
    
    try:
        # Validate input
        if not request.genres and not request.favorites:
            raise HTTPException(
                status_code=400, 
                detail="At least one genre or favorite movie must be provided"
            )
        
        # Limit top_n
        top_n = min(request.top_n, 50)
        
        user_vec = get_user_vector(request.genres, request.favorites)
        top_movies = get_recommendations(user_vec, top_n)
        
        return {
            "success": True,
            "user_input": {
                "genres": request.genres,
                "favorites": request.favorites,
                "requested_count": request.top_n,
                "returned_count": len(top_movies)
            },
            "recommendations": top_movies
        }
    except Exception as e:
        logger.error(f"Error in recommend_movies: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/movies/{movie_id}/recommendations")
async def get_similar_movies(movie_id: int, top_n: int = 10):
    """Get rekomendasi berdasarkan movie ID (similar movies)"""
    if not models_initialized:
        raise HTTPException(
            status_code=503, 
            detail=f"Models are not ready. Error: {initialization_error}"
        )
    
    try:
        top_n = min(top_n, 50)  # Limit maksimal
        recommendations = get_movie_recommendations_by_id(movie_id, top_n)
        
        if "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        
        return {
            "success": True,
            **recommendations
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_similar_movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint untuk monitoring"""
    return {
        "status": "healthy" if models_initialized else "unhealthy",
        "models_loaded": models_initialized,
        "total_movies": len(movies_df) if movies_df is not None else 0,
        "initialization_error": initialization_error,
        "timestamp": "2024-01-01T00:00:00Z"  # You can use datetime.now()
    }

@app.get("/test")
async def test_endpoint():
    """Test endpoint untuk debugging"""
    try:
        if not models_initialized:
            return {
                "status": "Models not initialized",
                "error": initialization_error
            }
        
        # Test dengan data sample
        test_genres = ["Action", "Adventure", "Sci-Fi"]
        test_favorites = ["Spider-Man", "Iron Man"]
        
        user_vec = get_user_vector(test_genres, test_favorites)
        recommendations = get_recommendations(user_vec, 3)
        
        return {
            "test": "✅ SUCCESS",
            "models_loaded": True,
            "sample_input": {
                "genres": test_genres,
                "favorites": test_favorites
            },
            "sample_recommendations": recommendations,
            "total_movies_loaded": len(movies_df)
        }
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {
            "test": "❌ FAILED",
            "error": str(e),
            "models_loaded": models_initialized
        }

# For Azure App Service
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    