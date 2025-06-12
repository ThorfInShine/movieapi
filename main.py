from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from utils import get_user_vector, get_recommendations

app = FastAPI()

class RecommendationRequest(BaseModel):
    genres: List[str]
    favorites: List[str]

@app.post("/recommend")
def recommend_movies(request: RecommendationRequest):
    user_vec = get_user_vector(request.genres, request.favorites)
    top_movies = get_recommendations(user_vec)
    return {"recommendations": top_movies}
