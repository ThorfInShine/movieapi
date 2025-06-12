# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import utils dengan error handling
try:
    from utils import initialize_models, get_user_vector, get_recommendations, movies_df
    logger.info("Utils imported successfully")
except Exception as e:
    logger.error(f"Failed to import utils: {e}")
    # Create dummy functions untuk prevent crash
    def initialize_models(): pass
    def get_user_vector(*args): return None
    def get_recommendations(*args): return []
    movies_df = None

app = FastAPI(
    title="Movie Recommendation API",
    description="API untuk rekomendasi film",
    version="1.0.0"
)

# CORS
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

# Global state
models_initialized = False
initialization_error = None

@app.on_event("startup")
async def startup_event():
    global models_initialized, initialization_error
    try:
        logger.info("🚀 Starting Movie API...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")
        
        # Check if models directory exists
        if os.path.exists('models'):
            logger.info(f"Models directory found: {os.listdir('models')}")
        else:
            logger.error("Models directory not found!")
            
        initialize_models()
        models_initialized = True
        logger.info("✅ Models initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        initialization_error = str(e)
        models_initialized = False

@app.get("/")
async def root():
    return {
        "message": "🎬 Movie Recommendation API",
        "status": "running" if models_initialized else "error",
        "models_loaded": models_initialized,
        "error": initialization_error,
        "python_version": sys.version,
        "working_dir": os.getcwd(),
        "endpoints": {
            "recommend": "POST /recommend",
            "health": "GET /health",
            "test": "GET /test",
            "docs": "GET /docs"
        }
    }

@app.post("/recommend")
async def recommend_movies(request: RecommendationRequest):
    if not models_initialized:
        raise HTTPException(
            status_code=503, 
            detail=f"Models not ready: {initialization_error}"
        )
    
    try:
        user_vec = get_user_vector(request.genres, request.favorites)
        recommendations = get_recommendations(user_vec, request.top_n)
        
        return {
            "success": True,
            "user_input": {
                "genres": request.genres,
                "favorites": request.favorites,
                "top_n": request.top_n
            },
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy" if models_initialized else "unhealthy",
        "models_loaded": models_initialized,
        "error": initialization_error,
        "total_movies": len(movies_df) if movies_df else 0
    }

@app.get("/test")
async def test():
    try:
        return {
            "test": "success",
            "models_initialized": models_initialized,
            "python_version": sys.version,
            "cwd": os.getcwd(),
            "files": os.listdir('.'),
            "models_files": os.listdir('models') if os.path.exists('models') else "No models dir"
        }
    except Exception as e:
        return {"test": "failed", "error": str(e)}

# For App Service
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)