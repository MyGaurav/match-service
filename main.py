from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define data structure for request
class MatchRequest(BaseModel):
    lost_text: str
    found_texts: List[str]

# Define structure for response
class MatchResponse(BaseModel):
    matches: List[float]

@app.post("/compare", response_model=MatchResponse)
def compare_posts(req: MatchRequest):
    """
    Takes one lost post and a list of found posts,
    returns cosine similarity scores.
    """
    try:
        # Encode text
        all_texts = [req.lost_text] + req.found_texts
        embeddings = model.encode(all_texts)

        # Compute cosine similarity between lost and each found
        lost_vec = embeddings[0].reshape(1, -1)
        found_vecs = embeddings[1:]

        scores = cosine_similarity(lost_vec, found_vecs)[0]

        return {"matches": [round(float(s), 4) for s in scores]}

    except Exception as e:
        return {"matches": [], "error": str(e)}
    