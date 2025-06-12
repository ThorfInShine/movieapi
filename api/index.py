# api/index.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_user_vector, get_recommendations

app = FastAPI(
    title="Movie Recommendation API",
    description="AI-powered movie recommendation system",
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
    
    class Config:
        schema_extra = {
            "example": {
                "genres": ["Action", "Sci-Fi"],
                "favorites": ["The Matrix", "Blade Runner"]
            }
        }

@app.get("/")
def root():
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "deployed on Vercel"
    }

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "platform": "Vercel"}

@app.post("/api/recommend")
def recommend_movies(request: RecommendationRequest):
    try:
        if not request.genres and not request.favorites:
            raise HTTPException(
                status_code=422,
                detail="Provide at least one genre or favorite movie"
            )
        
        user_vec = get_user_vector(request.genres, request.favorites)
        recommendations = get_recommendations(user_vec)
        
        return {
            "recommendations": recommendations,
            "status": "success",
            "count": len(recommendations),
            "input": {
                "genres": request.genres,
                "favorites": request.favorites
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# Export for Vercel
handler = app